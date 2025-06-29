"""
MultiModal RAG 시스템 유틸리티 클래스
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import time

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from loguru import logger

from config import Config
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

# 새로운 적응형 컴포넌트들 import
from adaptive_rag_components import (
    AdaptiveKeywordExtractor,
    AdaptiveWeightCalculator, 
    SmartThresholdCalculator,
    FlyWheelMetricsCollector,
    DomainDetector
)

# 캐싱을 위한 간단한 메모리 캐시 (Redis 대신 임시로 사용)
KEYWORD_CACHE = {}
DOMAIN_KEYWORDS_CACHE = {}
MAX_CACHE_SIZE = 1000

@dataclass
class Document:
    """문서 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    doc_type: str  # "text" or "image"

# DomainDetector와 AdaptiveKeywordExtractor는 adaptive_rag_components에서 import

class AnswerGenerator:
    """답변 생성 LLM 클래스"""
    
    def __init__(self, model_name: str = Config.LLM_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _check_tensor_health(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """텐서의 수치적 건전성을 확인합니다."""
        try:
            if tensor is None:
                logger.error(f"{name}이 None입니다")
                return False
            
            # NaN 확인
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                logger.error(f"{name}에 NaN 값이 {nan_count}개 있습니다")
                return False
            
            # Inf 확인
            if torch.isinf(tensor).any():
                inf_count = torch.isinf(tensor).sum().item()
                logger.error(f"{name}에 무한대 값이 {inf_count}개 있습니다")
                return False
            
            # 매우 큰 값 확인 (overflow 위험)
            max_val = torch.abs(tensor).max().item()
            if max_val > 1e6:
                logger.warning(f"{name}에 매우 큰 값이 있습니다: {max_val:.2e}")
                return False
            
            # 매우 작은 값 확인 (underflow 위험)
            min_val = torch.abs(tensor).min().item()
            if min_val > 0 and min_val < 1e-10:
                logger.warning(f"{name}에 매우 작은 값이 있습니다: {min_val:.2e}")
                # 작은 값은 경고만 하고 통과
            
            # 텐서 모양 확인
            if tensor.numel() == 0:
                logger.error(f"{name}이 비어있습니다")
                return False
                
            logger.debug(f"{name} 건전성 확인 완료 - 모양: {tensor.shape}, 범위: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
            return True
            
        except Exception as e:
            logger.error(f"{name} 건전성 확인 중 오류: {e}")
            return False
    
    def _safe_gpu_memory_cleanup(self):
        """안전한 GPU 메모리 정리"""
        try:
            if torch.cuda.is_available():
                # 모든 GPU에 대해 메모리 정리
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"GPU 메모리 정리 중 오류: {e}")
    
    def _monitor_gpu_memory(self, step: str = "unknown"):
        """GPU 메모리 사용량 모니터링"""
        if not torch.cuda.is_available():
            return
        
        try:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                max_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                if allocated > max_memory * 0.95:  # 95% 이상 사용 시 경고
                    logger.warning(f"[{step}] GPU {i} 메모리 부족 위험: {allocated:.1f}GB/{max_memory:.1f}GB")
                else:
                    logger.info(f"[{step}] GPU {i} 메모리: {allocated:.1f}GB 할당, {cached:.1f}GB 예약")
                    
        except Exception as e:
            logger.warning(f"GPU 메모리 모니터링 중 오류: {e}")
    
    def _load_model(self):
        """LLM 모델을 로드합니다."""
        try:
            logger.info(f"LLM 모델 로딩 중: {self.model_name}")
            
            # 토크나이저 로딩 (안전성 향상)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True  # 빠른 토크나이저 사용
            )
            
            # GPU 메모리 상황에 따른 조건부 로딩
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                total_memory = 0
                
                # 모든 GPU 메모리 확인
                for i in range(gpu_count):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    total_memory += gpu_memory
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - {gpu_memory:.1f}GB")
                
                logger.info(f"총 GPU 개수: {gpu_count}, 총 메모리: {total_memory:.1f}GB")
                
                # 공통 로딩 설정 (Gemma-3 호환)
                model_kwargs = {
                    "torch_dtype": torch.float16,  # 메모리 효율성
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                
                # Gemma 모델의 Float16 호환성 문제 해결
                if "gemma" in self.model_name.lower() and Config.FORCE_FLOAT32_FOR_GEMMA:
                    logger.warning("🔧 Gemma 모델 감지 - Float16 호환성 문제로 인해 Float32 사용")
                    model_kwargs["torch_dtype"] = torch.float32
                
                if gpu_count >= 2 and total_memory >= 40:  # 멀티 GPU & 충분한 메모리
                    logger.info("멀티 GPU 환경 - 자동 분산 설정 사용")
                    model_kwargs.update({
                        "device_map": "auto",  # 자동 분산으로 변경
                        "max_memory": {0: "22GB", 1: "22GB"}  # 각 GPU별 메모리 제한
                    })
                    
                    # Float32 사용 시 메모리 제한 조정
                    if model_kwargs["torch_dtype"] == torch.float32:
                        logger.info("Float32 사용으로 인한 메모리 제한 조정")
                        model_kwargs["max_memory"] = {0: "18GB", 1: "18GB"}
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    )
                elif gpu_count >= 1 and total_memory >= 32:  # 단일 GPU & 충분한 메모리
                    logger.info("충분한 GPU 메모리 - float16 최적 설정 사용")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory={0: "40GB"}
                    )
                elif gpu_count >= 1 and total_memory >= 40:  # 제한적 GPU 메모리
                    logger.info("제한적 GPU 메모리 - 최적화된 설정 사용")
                    gpu_memory_per_device = total_memory / gpu_count
                    max_memory_per_gpu = f"{gpu_memory_per_device-4:.0f}GB"
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory={i: max_memory_per_gpu for i in range(gpu_count)}
                    )
                else:  # 메모리 부족
                    logger.warning(f"GPU 메모리 부족 (총 {total_memory:.1f}GB) - CPU 사용 권장")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,  # CPU에서는 float32 사용
                        device_map="cpu",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            else:
                # CPU 전용
                logger.info("GPU 없음 - CPU 사용")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # CPU에서는 float32 사용
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            # 토크나이저 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # 모델 평가 모드 설정 (추론 최적화)
            self.model.eval()
            
            # 수치적 안정성을 위한 모델 후처리
            if hasattr(self.model.config, 'torch_dtype'):
                logger.info(f"모델 데이터 타입: {self.model.config.torch_dtype}")
            
            # 메모리 사용량 체크 (모든 GPU)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"GPU {i} 메모리 사용량 - 할당: {allocated:.1f}GB, 예약: {cached:.1f}GB")
            
            logger.info("LLM 모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"LLM 모델 로딩 실패: {e}")
            raise
    
    def generate_answer(self, query: str, context: List[str], sources: List[Dict] = None, 
                       has_relevant_docs: bool = True, max_length: int = 1024) -> str:
        """컨텍스트를 바탕으로 답변을 생성합니다."""
        import time
        start_time = time.time()
        
        try:
            if not has_relevant_docs or not context:
                # 문서와 연관 없는 질문에 대한 유도 답변
                return self._generate_guidance_answer(query)
            
            # 더 짧은 컨텍스트 구성 (토큰 제한 고려)
            context_with_sources = []
            total_length = 0
            max_context_length = 600  # 더 보수적으로 줄임
            
            for i, (ctx, source) in enumerate(zip(context[:2], sources[:2])):  # 최대 2개 문서만
                file_name = source.get('metadata', {}).get('file_name', f'문서{i+1}')
                # 각 컨텍스트를 250자로 제한 (더 짧게)
                truncated_ctx = ctx[:250] + "..." if len(ctx) > 250 else ctx
                doc_text = f"[{file_name}] {truncated_ctx}"
                
                if total_length + len(doc_text) > max_context_length:
                    break
                
                context_with_sources.append(doc_text)
                total_length += len(doc_text)
            
            context_text = "\n".join(context_with_sources)
            
            # 더욱 간단한 프롬프트
            prompt = f"문서: {context_text}\n질문: {query}\n답변:"
            
            logger.info(f"LLM에 전달될 프롬프트 길이: {len(prompt)}")
            logger.debug(f"프롬프트 시작 200자: '{prompt[:200]}'")
            
            try:
                # 매우 안전한 토큰화 설정
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=200,   # 더욱 짧게 설정
                    padding=False,
                    add_special_tokens=True
                )
                
                logger.debug(f"토큰화 결과 - input_ids shape: {inputs['input_ids'].shape}")
                
                # 입력 길이 검증 및 안전 장치
                input_length = inputs['input_ids'].shape[1]
                logger.debug(f"실제 입력 토큰 길이: {input_length}")
                
                # 빈 입력 검증
                if input_length < 5:
                    logger.error("입력이 너무 짧습니다.")
                    return self._create_context_summary(query, context, sources)
                
                # GPU 메모리 정리 및 텐서 이동
                self._monitor_gpu_memory("생성 전")
                self._safe_gpu_memory_cleanup()
                
                # 입력 텐서 건전성 확인
                if not self._check_tensor_health(inputs['input_ids'], "input_ids"):
                    logger.error("입력 텐서에 문제가 있습니다")
                    return self._create_context_summary(query, context, sources)
                
                # Gemma-3-12b-it 전용: 로짓 사전 검사
                if "gemma" in self.model_name.lower():
                    logger.info("🔍 Gemma 모델 감지 - 로짓 사전 검사 실행")
                    try:
                        with torch.no_grad():
                            # 로짓만 먼저 확인 (generate 전)
                            test_outputs = self.model(**inputs)
                            test_logits = test_outputs.logits
                            
                            # 로짓 상태 상세 로깅
                            # Float16 호환성을 위해 안전한 연산 사용
                            if test_logits.dtype == torch.float16:
                                logger.debug("Float16 로짓 감지 - float32로 변환하여 통계 계산")
                                logits_f32 = test_logits.float()
                                logits_min = logits_f32.min().item()
                                logits_max = logits_f32.max().item()
                                logits_mean = logits_f32.mean().item()
                                logits_std = logits_f32.std().item()
                            else:
                                logits_min = test_logits.min().item()
                                logits_max = test_logits.max().item()
                                logits_mean = test_logits.mean().item()
                                logits_std = test_logits.std().item()
                            
                            # NaN 체크 추가
                            if any(x != x for x in [logits_min, logits_max, logits_mean, logits_std]):  # NaN 체크
                                logger.error("🚨 로짓 통계에 NaN 값 발견!")
                                logger.debug(f"원본 로짓 dtype: {test_logits.dtype}, device: {test_logits.device}")
                                logger.debug(f"로짓 텐서 유효성: min_valid={torch.isfinite(test_logits).all()}")
                                risk_score += 3
                            else:
                                logger.info(f"📊 로짓 통계: 범위[{logits_min:.3f}, {logits_max:.3f}], 평균: {logits_mean:.3f}, 표준편차: {logits_std:.3f}")
                            
                            # 위험 신호 감지
                            risk_score = 0
                            if abs(logits_max) > 50:
                                logger.warning(f"⚠️ 로짓 최대값 위험: {logits_max:.3f}")
                                risk_score += 1
                            if abs(logits_min) < -50:
                                logger.warning(f"⚠️ 로짓 최소값 위험: {logits_min:.3f}")
                                risk_score += 1
                            if logits_std > 20:
                                logger.warning(f"⚠️ 로짓 분산 큼: {logits_std:.3f}")
                                risk_score += 1
                            
                            # 안정화된 softmax 테스트
                            try:
                                # Float16 호환성을 위해 float32로 캐스팅
                                if test_logits.dtype == torch.float16:
                                    logger.debug("Float16 감지 - float32로 캐스팅하여 softmax 테스트")
                                    test_logits_f32 = test_logits.float()
                                    stable_logits = test_logits_f32 - test_logits_f32.max(dim=-1, keepdim=True).values
                                    test_probs = torch.softmax(stable_logits, dim=-1)
                                else:
                                    stable_logits = test_logits - test_logits.max(dim=-1, keepdim=True).values
                                    test_probs = torch.softmax(stable_logits, dim=-1)
                                
                                if torch.isnan(test_probs).any():
                                    logger.error("❌ 안정화된 softmax에서 NaN 발생!")
                                    risk_score += 3
                                else:
                                    logger.debug("✅ Softmax 안정성 테스트 통과")
                                    
                            except Exception as softmax_error:
                                logger.error(f"❌ Softmax 테스트 실패: {softmax_error}")
                                # Float16 관련 오류인지 확인
                                if "Half" in str(softmax_error) or "float16" in str(softmax_error).lower():
                                    logger.warning("🔄 Float16 호환성 문제 감지 - 모델 정밀도 조정 필요")
                                risk_score += 3
                            
                            # 위험도 기반 생성 전략 결정
                            if risk_score >= 3:
                                logger.warning(f"🚨 높은 위험도 ({risk_score}) - 극도로 보수적 생성 모드")
                                generation_mode = "ultra_safe"
                            elif risk_score >= 1:
                                logger.warning(f"⚠️ 중간 위험도 ({risk_score}) - 안전 생성 모드")
                                generation_mode = "safe"
                            else:
                                logger.info("✅ 낮은 위험도 - 일반 생성 모드")
                                generation_mode = "normal"
                                
                            logger.info(f"🎯 선택된 생성 모드: {generation_mode}")
                            
                    except Exception as logits_check_error:
                        logger.error(f"❌ 로짓 사전 검사 실패: {logits_check_error}")
                        generation_mode = "ultra_safe"
                else:
                    generation_mode = "normal"
                
                # 텐서 디바이스 이동
                if torch.cuda.is_available() and inputs['input_ids'].device.type == 'cpu':
                    try:
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                        logger.debug("✅ 텐서를 GPU로 이동 완료")
                    except Exception as device_error:
                        logger.warning(f"❌ GPU로 텐서 이동 실패: {device_error}, CPU 모드 사용")

                with torch.no_grad():
                    # 수치 안정성을 위한 개선된 생성
                    try:
                        # 패딩 토큰 설정
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        
                        # 위험도별 생성 파라미터 설정
                        if generation_mode == "ultra_safe":
                            logger.info("🔒 극도로 보수적인 생성 설정 적용")
                            gen_kwargs = {
                                "input_ids": inputs['input_ids'],
                                "attention_mask": inputs['attention_mask'],
                                "max_new_tokens": 80,  # 30 → 80으로 증가
                                "do_sample": False,  # Greedy only
                                "pad_token_id": self.tokenizer.pad_token_id,
                                "eos_token_id": self.tokenizer.eos_token_id,
                                "early_stopping": True
                            }
                        elif generation_mode == "safe":
                            logger.info("🛡️ 안전한 생성 설정 적용")
                            gen_kwargs = {
                                "input_ids": inputs['input_ids'],
                                "attention_mask": inputs['attention_mask'],
                                "max_new_tokens": 100,  # 40 → 100으로 증가
                                "do_sample": True,
                                "temperature": 1.0,  # 안정적인 온도
                                "top_p": 0.95,
                                "top_k": 50,
                                "pad_token_id": self.tokenizer.pad_token_id,
                                "eos_token_id": self.tokenizer.eos_token_id,
                                "early_stopping": True
                            }
                        else:  # normal
                            logger.info("🌟 일반 생성 설정 적용")
                            gen_kwargs = {
                                "input_ids": inputs['input_ids'],
                                "attention_mask": inputs['attention_mask'],
                                "max_new_tokens": 120,  # 50 → 120으로 증가
                                "do_sample": True,
                                "temperature": 1.2,
                                "top_p": 0.9,
                                "top_k": 40,
                                "pad_token_id": self.tokenizer.pad_token_id,
                                "eos_token_id": self.tokenizer.eos_token_id,
                                "repetition_penalty": 1.02
                            }
                        
                        # 첫 번째 시도
                        logger.debug("첫 번째 생성 시도 시작")
                        logger.debug(f"생성 파라미터: {gen_kwargs}")
                        outputs = self.model.generate(**gen_kwargs)
                        logger.debug("첫 번째 생성 시도 성공")
                        
                    except Exception as greedy_error:
                        logger.warning(f"첫 번째 생성 시도 실패: {greedy_error}")
                        
                        # 오류 상세 분석
                        if "probability tensor" in str(greedy_error):
                            logger.error("🚨 확률 텐서 수치 불안정성 감지!")
                            logger.debug("문제 해결 시도: 로짓 안정화 적용")
                        elif "cuda" in str(greedy_error).lower():
                            logger.error("🎮 CUDA 관련 오류 감지")
                        elif "memory" in str(greedy_error).lower():
                            logger.error("💾 메모리 관련 오류 감지")
                        
                        # GPU 메모리 정리
                        self._safe_gpu_memory_cleanup()
                        
                        try:
                            # 두 번째 시도: 극도로 단순한 설정
                            logger.debug("두 번째 생성 시도 시작 (Greedy only)")
                            fallback_kwargs = {
                                "input_ids": inputs['input_ids'],
                                "max_new_tokens": 60,  # 25 → 60으로 증가
                                "do_sample": False,     # Greedy only
                                "pad_token_id": self.tokenizer.pad_token_id,
                                "eos_token_id": self.tokenizer.eos_token_id
                            }
                            logger.debug(f"백업 생성 파라미터: {fallback_kwargs}")
                            outputs = self.model.generate(**fallback_kwargs)
                            logger.debug("두 번째 생성 시도 성공")
                            
                        except Exception as second_error:
                            logger.error(f"두 번째 생성 시도도 실패: {second_error}")
                            logger.error(f"모델 상태 - device: {next(self.model.parameters()).device}, dtype: {next(self.model.parameters()).dtype}")
                            
                            # 최종 백업: 컨텍스트 기반 요약으로 대체
                            logger.info("🔄 모든 생성 시도 실패 - 컨텍스트 요약으로 대체")
                            return self._create_context_summary(query, context, sources)
                
                # 생성된 텍스트 추출 (더 안전한 방식)
                try:
                    input_ids = inputs['input_ids'][0]
                    output_ids = outputs[0]
                    
                    # 입력 토큰 제거하여 생성된 부분만 추출
                    generated_ids = output_ids[len(input_ids):]
                    
                    # 디버깅을 위한 상세 로깅
                    logger.debug(f"입력 토큰 길이: {len(input_ids)}, 출력 토큰 길이: {len(output_ids)}")
                    logger.debug(f"생성된 토큰 길이: {len(generated_ids)}")
                    
                    if len(generated_ids) > 0:
                        logger.debug(f"생성된 첫 5개 토큰: {generated_ids[:5].tolist()}")
                        
                        # 패딩 토큰과 0 토큰 제거
                        clean_tokens = []
                        for token in generated_ids:
                            token_id = token.item()
                            if token_id not in [0, self.tokenizer.pad_token_id]:
                                clean_tokens.append(token)
                            elif token_id == self.tokenizer.eos_token_id:
                                break  # EOS 토큰에서 중지
                        
                        if clean_tokens:
                            clean_tensor = torch.stack(clean_tokens)
                            raw_answer = self.tokenizer.decode(clean_tensor, skip_special_tokens=True).strip()
                            logger.debug(f"정리된 토큰 개수: {len(clean_tokens)}")
                        else:
                            raw_answer = ""
                            logger.warning("정리된 토큰이 없습니다")
                    else:
                        raw_answer = ""
                        logger.warning("생성된 토큰이 없습니다!")
                    
                    logger.info(f"LLM 원본 답변: '{raw_answer}' (길이: {len(raw_answer)})")
                    
                    # 답변 후처리
                    answer = self._clean_generated_answer(raw_answer, query) if raw_answer else ""
                    
                    # 최소 길이 체크
                    if not answer or len(answer.strip()) < 5:
                        logger.warning(f"생성된 답변이 너무 짧음: '{answer}'")
                        # 컨텍스트 기반 요약 답변으로 대체
                        return self._create_context_summary(query, context, sources)
                    
                    elapsed_time = time.time() - start_time
                    
                    # 답변 품질 메트릭 수집 (Gemma 모델 전용)
                    if "gemma" in self.model_name.lower() and answer:
                        self._log_answer_quality_metrics(raw_answer, answer, generated_ids, elapsed_time)
                    logger.info(f"LLM 답변 생성 성공 (소요시간: {elapsed_time:.2f}초), 답변 길이: {len(answer)}")
                    
                    # 성공 메트릭 로깅
                    self._log_success_metrics(query, answer, elapsed_time, generation_mode if 'generation_mode' in locals() else 'unknown')
                    
                    self._monitor_gpu_memory("생성 완료")
                    
                    return answer
                    
                except Exception as extraction_error:
                    logger.error(f"텍스트 추출 중 오류: {extraction_error}")
                    return self._create_context_summary(query, context, sources)
                
            except Exception as tokenization_error:
                logger.error(f"토큰화 중 오류: {tokenization_error}")
                return self._create_context_summary(query, context, sources)
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"답변 생성 실패 (소요시간: {elapsed_time:.2f}초): {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _clean_generated_answer(self, answer: str, query: str) -> str:
        """생성된 답변을 정리하고 개선합니다."""
        if not answer:
            return ""
        
        # 불필요한 패턴 제거
        patterns_to_remove = [
            "답변:", "응답:", "Answer:", "Response:",
            "질문:", "Question:", "Q:", "A:",
            "참고 문서:", "참고:", "출처:",
            "[출처:", "한국어 답변:", "답변은 다음과 같습니다:",
            "다음과 같이 답변드립니다:", "답변드리겠습니다:"
        ]
        
        cleaned_answer = answer
        for pattern in patterns_to_remove:
            if pattern in cleaned_answer:
                parts = cleaned_answer.split(pattern, 1)
                if len(parts) > 1 and len(parts[1].strip()) > 20:
                    cleaned_answer = parts[1].strip()
        
        # 반복적인 문장 제거
        sentences = cleaned_answer.split('.')
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                # 문장 유사성 검사 (간단한 버전)
                sentence_key = sentence.lower()[:50]  # 첫 50자로 중복 검사
                if sentence_key not in seen_sentences:
                    unique_sentences.append(sentence)
                    seen_sentences.add(sentence_key)
        
        cleaned_answer = '. '.join(unique_sentences)
        if cleaned_answer and not cleaned_answer.endswith('.'):
            cleaned_answer += '.'
        
        # 최종 검증
        if len(cleaned_answer.strip()) < 20:
            return answer  # 원본 반환
        
        return cleaned_answer.strip()
    
    def _create_context_summary(self, query: str, context: List[str], sources: List[Dict]) -> str:
        """컨텍스트를 기반으로 LLM 없이 요약 답변을 생성합니다."""
        try:
            # 질문 키워드 추출
            query_lower = query.lower()
            
            # 질문 유형별 맞춤 답변 생성
            if "소비자물가" in query or "cpi" in query_lower or "인플레이션" in query_lower:
                # 소비자물가 관련 특화 답변
                answer = self._generate_cpi_summary(context, sources)
            elif "고용" in query or "실업" in query or "일자리" in query:
                # 고용 관련 특화 답변
                answer = self._generate_employment_summary(context, sources)
            else:
                # 일반적인 경제 분석 답변
                answer = self._generate_general_summary(query, context, sources)
            
            return answer
            
        except Exception as e:
            logger.error(f"컨텍스트 요약 생성 실패: {e}")
            return f"죄송합니다. '{query}' 질문에 대한 답변을 생성할 수 없습니다. 참고 문서를 직접 확인해주세요."
    
    def _generate_cpi_summary(self, context: List[str], sources: List[Dict]) -> str:
        """소비자물가 관련 특화 요약 생성 (키워드 기반)"""
        return self._generate_keyword_based_summary("소비자물가", ["CPI", "소비자물가", "인플레이션", "물가"], context, sources)
    
    def _generate_employment_summary(self, context: List[str], sources: List[Dict]) -> str:
        """고용 관련 특화 요약 생성 (키워드 기반)"""
        return self._generate_keyword_based_summary("미국 고용지표", ["고용", "실업", "일자리", "취업", "임금", "실업률"], context, sources)
    
    def _generate_general_summary(self, query: str, context: List[str], sources: List[Dict]) -> str:
        """일반적인 경제 분석 요약 생성 (고급 키워드 기반)"""
        # 질문에서 키워드 추출 (컨텍스트 활용)
        query_keywords = self._extract_query_keywords(query, context)
        
        if query_keywords:
            return self._generate_keyword_based_summary(query, query_keywords, context, sources)
        else:
            # 기본 방식으로 폴백 (개선된 버전)
            key_points = []
            for i, (ctx, source) in enumerate(zip(context[:3], sources[:3])):
                file_name = source.get('metadata', {}).get('file_name', f'문서{i+1}')
                
                # 더 의미있는 요약 생성
                sentences = ctx.split('.')[:3]  # 첫 3문장
                summary = '. '.join(sentence.strip() for sentence in sentences if len(sentence.strip()) > 10)
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                
                key_points.append(f"• **{file_name}**: {summary}")
            
            return f""""{query}"에 대한 정보를 찾았습니다.

📋 **주요 내용**:
{chr(10).join(key_points)}

💡 **참고**: 더 정확한 분석을 위해 구체적인 키워드(예: 'CPI', '실업률', '금리' 등)를 포함한 질문을 해보시기 바랍니다.

검색된 문서들에는 한국은행이 분석한 최신 데이터를 바탕으로 한 것입니다."""
    
    def _extract_query_keywords(self, query: str, context: List[str] = None) -> List[str]:
        """
        고급 키워드 추출 시스템을 사용하여 질문에서 핵심 키워드를 추출합니다.
        """
        if not hasattr(self, 'keyword_extractor'):
            # 첫 번째 호출 시 키워드 추출기 초기화
            embedding_model = EmbeddingModel()
            self.keyword_extractor = AdaptiveKeywordExtractor(embedding_model)
        
        # 고급 키워드 추출 수행
        extraction_result = self.keyword_extractor.extract_keywords(query, context)
        
        # 상세 분석 정보 저장 (디버깅 및 추후 사용)
        self._last_keyword_analysis = extraction_result
        
        # 로그로 키워드 추출 과정 기록
        logger.info(f"키워드 추출 완료 - 질문: {query[:30]}...")
        logger.info(f"  형태소 분석: {extraction_result['morphological_keywords']}")
        logger.info(f"  의미적 확장: {extraction_result['semantic_keywords']}")
        logger.info(f"  TF-IDF: {extraction_result['tfidf_keywords']}")
        logger.info(f"  최종 키워드: {extraction_result['final_keywords']}")
        
        return extraction_result["final_keywords"]
    
    def _generate_keyword_based_summary(self, topic: str, keywords: List[str], context: List[str], sources: List[Dict]) -> str:
        """
        고급 키워드 시스템을 활용한 정교한 컨텍스트 분석 및 답변 생성
        """
        relevant_passages = []
        
        # 키워드 분석 정보 활용 (있는 경우)
        if hasattr(self, '_last_keyword_analysis'):
            analysis = self._last_keyword_analysis
            all_keywords = analysis.get('final_keywords', keywords)
            morphological_keywords = analysis.get('morphological_keywords', [])
            semantic_keywords = analysis.get('semantic_keywords', [])
        else:
            all_keywords = keywords
            morphological_keywords = keywords
            semantic_keywords = []
        
        for ctx in context[:3]:
            # 정확한 문장 분리 (형태소 분석기가 있다면 활용)
            if hasattr(self, 'keyword_extractor'):
                sentences = self.keyword_extractor._split_sentences_accurately(ctx)
            else:
                sentences = [s.strip() for s in ctx.split('.') if s.strip()]
            
            keyword_sentences = []
            
            for i, sentence in enumerate(sentences):
                if len(sentence) < 10:
                    continue
                
                # 다층적 키워드 매칭
                sentence_score = 0
                
                # 1. 형태소 분석 키워드 매칭 (가중치 1.0)
                for keyword in morphological_keywords:
                    if keyword in sentence:
                        sentence_score += 1.0
                
                # 2. 의미적 확장 키워드 매칭 (가중치 0.8)
                for keyword in semantic_keywords:
                    if keyword in sentence:
                        sentence_score += 0.8
                
                # 3. 전체 키워드 매칭 (가중치 0.6)
                for keyword in all_keywords:
                    if keyword in sentence and keyword not in morphological_keywords and keyword not in semantic_keywords:
                        sentence_score += 0.6
                
                # 임계값 이상인 문장만 선택
                if sentence_score >= 0.8:
                    # 앞뒤 문맥 포함하여 완전한 정보 구성
                    start_idx = max(0, i - 1)
                    end_idx = min(len(sentences), i + 2)
                    
                    context_sentence = " ".join(sentences[start_idx:end_idx])
                    
                    # 중복 방지 및 품질 검증
                    if (context_sentence not in keyword_sentences and 
                        len(context_sentence) > 20 and
                        any(keyword in context_sentence for keyword in all_keywords)):
                        keyword_sentences.append((context_sentence, sentence_score))
            
            if keyword_sentences:
                # 점수 순으로 정렬하여 상위 2개만 선택
                keyword_sentences.sort(key=lambda x: x[1], reverse=True)
                relevant_passages.extend([passage for passage, score in keyword_sentences[:2]])
        
        # 중복 제거 및 품질 필터링
        unique_passages = []
        for passage in relevant_passages:
            # 의미있는 내용인지 확인
            if (passage not in unique_passages and 
                len(passage) > 30 and
                any(keyword in passage for keyword in all_keywords)):
                unique_passages.append(passage)
        
        if unique_passages:
            # 키워드 기반 고품질 답변 생성
            passages_text = "\n\n".join([f"📊 {passage}" for passage in unique_passages[:3]])
            
            # 고급 트렌드 분석 (키워드 분석 정보 활용)
            combined_text = " ".join(unique_passages)
            trend_analysis = self._analyze_trend_from_text_advanced(combined_text, all_keywords)
            
            # 키워드 카테고리 분석 추가
            category_info = self._get_keyword_category_info(all_keywords)
            
            return f"""{topic} 관련 주요 정보:

{passages_text}

{trend_analysis}

{category_info}

위 정보는 한국은행이 분석한 최신 데이터를 바탕으로 한 것입니다."""
        
        else:
            # 키워드를 찾지 못한 경우에도 개선된 답변
            return f"""{topic}에 대한 정보가 검색되었으나, 구체적인 관련 내용을 찾기 어려웠습니다.

💡 **검색 개선 제안:**
- 더 구체적인 키워드 사용 (예: "소비자물가상승률", "고용지표", "금리인상" 등)
- 특정 시기나 지역 명시
- 경제지표의 정확한 명칭 사용

검색된 문서들에는 관련 분석과 데이터가 포함되어 있으니, 아래 참고 문서를 확인해보시기 바랍니다."""

    def _analyze_trend_from_text_advanced(self, text: str, keywords: List[str]) -> str:
        """키워드 정보를 활용한 고급 트렌드 분석"""
        if not text:
            return ""
        
        # 키워드별 가중치 적용 트렌드 분석
        trend_keywords = {
            "상승": ["상승", "증가", "확대", "강화", "개선", "호조", "급증"],
            "하락": ["하락", "감소", "축소", "악화", "부진", "하회", "급감"],
            "안정": ["안정", "유지", "동결", "보합", "횡보"],
            "변동": ["변동", "등락", "혼조", "변화"]
        }
        
        trend_scores = {}
        
        for trend_type, trend_words in trend_keywords.items():
            score = 0
            for word in trend_words:
                if word in text:
                    # 키워드 주변 문맥도 고려
                    count = text.count(word)
                    # 핵심 키워드와 함께 나타나면 가중치 추가
                    for keyword in keywords:
                        if keyword in text and abs(text.find(keyword) - text.find(word)) < 50:
                            score += count * 1.5
                        else:
                            score += count
            trend_scores[trend_type] = score
        
        # 가장 높은 점수의 트렌드 반환
        if not any(trend_scores.values()):
            return "📋 **분석**: 현재 데이터로는 명확한 트렌드를 파악하기 어렵습니다."
        
        dominant_trend = max(trend_scores, key=trend_scores.get)
        
        trend_messages = {
            "상승": "📈 **분석**: 전반적으로 상승세 또는 개선 흐름을 보이고 있습니다.",
            "하락": "📉 **분석**: 전반적으로 하락세 또는 둔화 흐름을 보이고 있습니다.",
            "안정": "📊 **분석**: 상대적으로 안정적인 흐름을 유지하고 있습니다.",
            "변동": "🔄 **분석**: 다양한 변동 요인들이 복합적으로 작용하고 있습니다."
        }
        
        return trend_messages.get(dominant_trend, "📋 **분석**: 다양한 요인들이 복합적으로 작용하고 있는 상황입니다.")
    
    def _get_keyword_category_info(self, keywords: List[str]) -> str:
        """키워드 카테고리 정보 제공"""
        if hasattr(self, '_last_keyword_analysis'):
            categories = self._last_keyword_analysis.get('analysis', {}).get('dominant_categories', {})
            
            category_names = {
                "물가": "💰 물가",
                "고용": "👥 고용", 
                "금융": "🏦 금융",
                "시장": "📈 시장",
                "경제": "🌍 경제",
                "수치": "📊 수치"
            }
            
            active_categories = []
            for category, count in categories.items():
                if count > 0:
                    active_categories.append(f"{category_names.get(category, category)}: {count}개 키워드")
            
            if active_categories:
                return f"🔍 **키워드 분야**: {', '.join(active_categories)}"
        
        return ""
    
    def _analyze_trend_from_text(self, text: str) -> str:
        """텍스트에서 트렌드를 분석합니다."""
        if not text:
            return ""
        
        # 긍정적/상승 키워드
        positive_keywords = ["상승", "증가", "개선", "강화", "확대", "호조"]
        # 부정적/하락 키워드  
        negative_keywords = ["하락", "감소", "악화", "축소", "부진", "하회"]
        # 중성/안정 키워드
        neutral_keywords = ["안정", "유지", "동결", "보합"]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text)
        neutral_count = sum(1 for keyword in neutral_keywords if keyword in text)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return "📈 **분석**: 전반적으로 상승세 또는 개선 흐름을 보이고 있습니다."
        elif negative_count > positive_count and negative_count > neutral_count:
            return "📉 **분석**: 전반적으로 하락세 또는 둔화 흐름을 보이고 있습니다."
        elif neutral_count > 0:
            return "📊 **분석**: 상대적으로 안정적인 흐름을 유지하고 있습니다."
        else:
            return "📋 **분석**: 다양한 요인들이 복합적으로 작용하고 있는 상황입니다."

    def _generate_guidance_answer(self, query: str) -> str:
        """문서와 연관 없는 질문에 대한 유도 답변을 생성합니다."""
        guidance_answer = f"""죄송합니다. 질문 "{query}"은 현재 시스템에 저장된 한국은행 뉴스 데이터와 직접적인 연관이 없는 것 같습니다.

🏦 **본 시스템에서 제공 가능한 정보:**
- 미국 소비자물가 동향 및 금융시장 반응
- 미국 고용지표 내용 및 뉴욕 금융시장 반응  
- 각종 경제지표 분석 및 해석
- 금융시장 동향 및 전망
- 통화정책 관련 정보

💡 **추천 질문 예시:**
- "미국 소비자물가 상승률은 어떻게 변화하고 있나요?"
- "최근 미국 고용지표가 금융시장에 미친 영향은 무엇인가요?"
- "한국은행이 분석한 미국 경제 동향은 어떠한가요?"
- "CPI 상승률이 금융시장에 미치는 영향을 설명해주세요."
- "미국 연방준비제도의 통화정책 변화는 어떤가요?"

위와 같은 한국은행 뉴스와 관련된 금융 질문을 해주시면 정확한 데이터를 바탕으로 자세한 답변을 제공해드릴 수 있습니다."""
        
        return guidance_answer

    def _log_answer_quality_metrics(self, raw_answer: str, cleaned_answer: str, generated_ids: torch.Tensor, elapsed_time: float):
        """답변 품질 관련 상세 메트릭을 로깅합니다."""
        try:
            # 기본 메트릭
            raw_length = len(raw_answer)
            cleaned_length = len(cleaned_answer)
            token_count = len(generated_ids)
            
            # 텍스트 품질 지표
            sentence_count = len([s for s in cleaned_answer.split('.') if s.strip()])
            word_count = len(cleaned_answer.split())
            avg_word_length = sum(len(word) for word in cleaned_answer.split()) / max(word_count, 1)
            
            # 토큰 효율성
            chars_per_token = raw_length / max(token_count, 1)
            tokens_per_second = token_count / max(elapsed_time, 0.001)
            
            # 정리 효과
            cleaning_ratio = cleaned_length / max(raw_length, 1)
            
            logger.info(f"📊 답변 품질 메트릭:")
            logger.info(f"   📝 길이: 원본 {raw_length}자 → 정리후 {cleaned_length}자 (정리율: {cleaning_ratio:.2f})")
            logger.info(f"   🔤 토큰: {token_count}개, 문장: {sentence_count}개, 단어: {word_count}개")
            logger.info(f"   📈 효율성: {chars_per_token:.1f}자/토큰, {tokens_per_second:.1f}토큰/초")
            logger.info(f"   📏 평균 단어 길이: {avg_word_length:.1f}자")
            
            # 토큰 분포 분석 (처음/끝 몇 개)
            if len(generated_ids) > 0:
                first_tokens = generated_ids[:3].tolist()
                last_tokens = generated_ids[-3:].tolist()
                logger.debug(f"🎯 토큰 패턴: 시작 {first_tokens} ... 끝 {last_tokens}")
                
                # 0 토큰 비율 (패딩 체크)
                zero_token_ratio = (generated_ids == 0).sum().item() / len(generated_ids)
                if zero_token_ratio > 0:
                    logger.warning(f"⚠️ 0 토큰 비율: {zero_token_ratio:.2%}")
            
        except Exception as e:
            logger.debug(f"품질 메트릭 로깅 중 오류: {e}")

    def _log_success_metrics(self, query: str, answer: str, elapsed_time: float, generation_mode: str):
        """성공적인 답변 생성에 대한 메트릭을 로깅합니다."""
        try:
            # 질문-답변 매칭도 간단 분석
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            word_overlap = len(query_words & answer_words) / max(len(query_words), 1)
            
            # 답변 특성
            is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in answer)
            has_numbers = any(char.isdigit() for char in answer)
            has_punctuation = any(char in '.,!?;:' for char in answer)
            
            # 성능 카테고리
            if elapsed_time < 2:
                speed_category = "매우빠름"
            elif elapsed_time < 5:
                speed_category = "빠름"
            elif elapsed_time < 10:
                speed_category = "보통"
            else:
                speed_category = "느림"
            
            logger.info(f"✅ 성공 메트릭 요약:")
            logger.info(f"   🚀 성능: {speed_category} ({elapsed_time:.2f}초)")
            logger.info(f"   🎯 생성모드: {generation_mode}")
            logger.info(f"   🔍 단어 겹침: {word_overlap:.2%}")
            logger.info(f"   🌏 언어특성: 한국어={is_korean}, 숫자={has_numbers}, 구두점={has_punctuation}")
            
            # 성공 패턴 추적을 위한 구조화된 로그
            success_data = {
                "timestamp": time.time(),
                "model": self.model_name,
                "generation_mode": generation_mode,
                "elapsed_time": elapsed_time,
                "query_length": len(query),
                "answer_length": len(answer),
                "word_overlap": word_overlap,
                "speed_category": speed_category
            }
            logger.debug(f"📊 성공 데이터: {success_data}")
            
        except Exception as e:
            logger.debug(f"성공 메트릭 로깅 중 오류: {e}")

    def _log_model_state(self, step: str = "unknown"):
        """모델 상태를 로깅합니다."""
        try:
            logger.debug(f"[{step}] 모델 상태 체크")
            
            # 모델 디바이스 확인
            model_device = next(self.model.parameters()).device
            logger.debug(f"[{step}] 모델 디바이스: {model_device}")
            
            # 모델 데이터 타입 확인
            model_dtype = next(self.model.parameters()).dtype
            logger.debug(f"[{step}] 모델 데이터 타입: {model_dtype}")
            
            # 모델 평가 모드 확인
            logger.debug(f"[{step}] 모델 평가 모드: {not self.model.training}")
            
            # 파라미터 건전성 간단 체크
            params = list(self.model.parameters())
            if params:
                first_param = params[0]
                if torch.isnan(first_param).any() or torch.isinf(first_param).any():
                    logger.error(f"[{step}] 모델 파라미터에 NaN 또는 Inf 값 발견!")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"[{step}] 모델 상태 확인 중 오류: {e}")
            return False

class VectorStore:
    """Qdrant 벡터 스토어 클래스"""
    
    def __init__(self, collection_name: str = Config.COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = QdrantClient(url=Config.QDRANT_URL)
        self._setup_collection()
    
    def _setup_collection(self):
        """컬렉션을 설정합니다."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1024,  # BGE-M3 모델의 임베딩 차원
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"컬렉션 '{self.collection_name}' 생성 완료")
            else:
                logger.info(f"컬렉션 '{self.collection_name}' 이미 존재")
                
        except Exception as e:
            logger.error(f"컬렉션 설정 실패: {e}")
            raise
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """문서들을 벡터 스토어에 추가합니다."""
        try:
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point = PointStruct(
                    id=len(points),
                    vector=embedding.tolist(),
                    payload={
                        "content": doc.content,
                        "doc_id": doc.doc_id,
                        "doc_type": doc.doc_type,
                        **doc.metadata
                    }
                )
                points.append(point)
            
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"{len(documents)}개 문서 벡터 스토어 추가 완료")
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, section_types: List[str] = None) -> List[Dict]:
        """유사한 문서들을 검색합니다."""
        try:
            # section_type 필터 설정 (기본값: analysis와 image_summary만)
            if section_types is None:
                section_types = ["analysis", "image_summary"]
            
            # 필터 조건 생성
            filter_condition = {
                "must": [
                    {
                        "key": "section_type",
                        "match": {
                            "any": section_types
                        }
                    }
                ]
            }
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True,
                score_threshold=Config.SIMILARITY_THRESHOLD,
                query_filter=filter_condition
            )
            
            return [
                {
                    "content": result.payload["content"],
                    "score": result.score,
                    "metadata": {k: v for k, v in result.payload.items() if k != "content"}
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []
    
    def search_questions(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """질문 섹션만 검색합니다."""
        return self.search(query_embedding, top_k, section_types=["questions"])

class DocumentLoader:
    """마크다운 문서 로더 클래스"""
    
    @staticmethod
    def load_markdown_files(directory: Path, doc_type: str) -> List[Document]:
        """마크다운 파일들을 로드하고 필요한 섹션만 추출합니다."""
        documents = []
        
        if not directory.exists():
            logger.warning(f"디렉토리가 존재하지 않습니다: {directory}")
            return documents
        
        # .md 파일만 선택 (Zone.Identifier 등 제외)
        markdown_files = [f for f in directory.glob("*.md") 
                         if not f.name.endswith(".mdZone.Identifier")]
        logger.info(f"{directory}에서 {len(markdown_files)}개 마크다운 파일 발견")
        
        for file_path in markdown_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if doc_type == "text":
                    # 텍스트 분석 결과에서 필요한 섹션 추출
                    extracted_docs = DocumentLoader._extract_text_analysis_sections(content, file_path)
                elif doc_type == "image":
                    # 이미지 분석 결과에서 필요한 섹션 추출
                    extracted_docs = DocumentLoader._extract_image_analysis_sections(content, file_path)
                else:
                    logger.warning(f"알 수 없는 doc_type: {doc_type}")
                    continue
                
                documents.extend(extracted_docs)
                
            except Exception as e:
                logger.error(f"파일 로딩 실패 {file_path}: {e}")
                continue
        
        logger.info(f"{len(documents)}개 문서 로딩 완료")
        return documents
    
    @staticmethod
    def _extract_text_analysis_sections(content: str, file_path: Path) -> List[Document]:
        """텍스트 분석 결과에서 🔍 분석 결과와 ❓ Hypothetical Questions 추출"""
        documents = []
        
        # 각 텍스트 분석 블록을 분리
        sections = content.split("## 📝 텍스트 분석")
        
        for i, section in enumerate(sections[1:], 1):  # 첫 번째는 헤더이므로 제외
            try:
                # 분석 결과 추출
                analysis_start = section.find("### 🔍 분석 결과")
                questions_start = section.find("### ❓ Hypothetical Questions")
                
                if analysis_start != -1 and questions_start != -1:
                    # 분석 결과 추출
                    analysis_content = section[analysis_start:questions_start].strip()
                    analysis_content = analysis_content.replace("### 🔍 분석 결과", "").strip()
                    
                    if analysis_content:
                        doc_id = f"{file_path.stem}_analysis_{i}"
                        metadata = {
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "section_type": "analysis",
                            "section_number": i
                        }
                        
                        documents.append(Document(
                            content=analysis_content,
                            metadata=metadata,
                            doc_id=doc_id,
                            doc_type="text"
                        ))
                    
                    # Hypothetical Questions 추출
                    next_section = section.find("---", questions_start)
                    if next_section == -1:
                        next_section = len(section)
                    
                    questions_content = section[questions_start:next_section].strip()
                    questions_content = questions_content.replace("### ❓ Hypothetical Questions", "").strip()
                    
                    if questions_content:
                        doc_id = f"{file_path.stem}_questions_{i}"
                        metadata = {
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "section_type": "questions",
                            "section_number": i
                        }
                        
                        documents.append(Document(
                            content=questions_content,
                            metadata=metadata,
                            doc_id=doc_id,
                            doc_type="text"
                        ))
                        
            except Exception as e:
                logger.error(f"텍스트 섹션 추출 실패 (섹션 {i}): {e}")
                continue
        
        return documents
    
    @staticmethod
    def _extract_image_analysis_sections(content: str, file_path: Path) -> List[Document]:
        """이미지 분석 결과에서 📋 Image Summary와 ❓ Hypothetical Questions 추출"""
        documents = []
        
        # 각 이미지 분석 블록을 분리
        sections = content.split("## 🖼️ 이미지")
        
        for i, section in enumerate(sections[1:], 1):  # 첫 번째는 헤더이므로 제외
            try:
                # 섹션에서 파일명 추출
                image_filename = ""
                lines = section.split('\n')
                for line in lines[:5]:  # 처음 5줄에서 파일명 찾기
                    if '.png' in line and '이미지' in line:
                        # "## 🖼️ 이미지 1: 파일명.png" 형태에서 파일명 추출
                        parts = line.split(': ')
                        if len(parts) > 1:
                            image_filename = parts[1].strip()
                        break
                
                # 파일명을 기반으로 이미지 타입 판별 (table vs picture)
                image_type = "table" if "-table-" in image_filename else "picture"
                
                # Image Summary 추출
                summary_start = section.find("### 📋 Image Summary")
                questions_start = section.find("### ❓ Hypothetical Questions")
                
                if summary_start != -1 and questions_start != -1:
                    # Image Summary 추출
                    summary_content = section[summary_start:questions_start].strip()
                    summary_content = summary_content.replace("### 📋 Image Summary", "").strip()
                    
                    if summary_content:
                        doc_id = f"{file_path.stem}_summary_{image_type}_{i}"
                        metadata = {
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "section_type": "image_summary",
                            "image_type": image_type,  # "table" or "picture"
                            "image_filename": image_filename,  # 실제 이미지 파일명
                            "section_number": i
                        }
                        
                        documents.append(Document(
                            content=summary_content,
                            metadata=metadata,
                            doc_id=doc_id,
                            doc_type="image"
                        ))
                    
                    # Hypothetical Questions 추출
                    next_section = section.find("---", questions_start)
                    if next_section == -1:
                        next_section = len(section)
                    
                    questions_content = section[questions_start:next_section].strip()
                    questions_content = questions_content.replace("### ❓ Hypothetical Questions", "").strip()
                    
                    if questions_content:
                        doc_id = f"{file_path.stem}_questions_{image_type}_{i}"
                        metadata = {
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "section_type": "questions",
                            "image_type": image_type,  # "table" or "picture"
                            "image_filename": image_filename,  # 실제 이미지 파일명
                            "section_number": i
                        }
                        
                        documents.append(Document(
                            content=questions_content,
                            metadata=metadata,
                            doc_id=doc_id,
                            doc_type="image"
                        ))
                        
            except Exception as e:
                logger.error(f"이미지 섹션 추출 실패 (섹션 {i}): {e}")
                continue
        
        return documents

class RAGSystem:
    """개선된 RAG 시스템 클래스 - 적응형 컴포넌트 통합"""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.reranker = RerankerModel()
        self.answer_generator = AnswerGenerator()
        self.vector_store = VectorStore()
        
        # 새로운 적응형 컴포넌트들 초기화
        self.adaptive_keyword_extractor = AdaptiveKeywordExtractor(self.embedding_model)
        self.adaptive_weight_calculator = AdaptiveWeightCalculator()
        self.smart_threshold_calculator = SmartThresholdCalculator()
        self.flywheel_metrics = FlyWheelMetricsCollector()
        self.domain_detector = DomainDetector()
        
        logger.info("개선된 RAG 시스템 초기화 완료 (적응형 컴포넌트 포함)")
        
    def search_and_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """개선된 질문 처리: 도메인 감지 + 적응형 키워드 + 스마트 임계값"""
        try:
            # 1. 도메인 자동 감지
            detected_domain = self.domain_detector.detect_domain(query)
            logger.debug(f"감지된 도메인: {detected_domain}")
            
            # 2. 적응형 키워드 추출
            keyword_result = self.adaptive_keyword_extractor.extract_keywords_adaptive(
                query, domain=detected_domain
            )
            logger.debug(f"추출된 키워드: {keyword_result['final_keywords']}")
            
            # 3. 쿼리 임베딩
            query_embedding = self.embedding_model.encode([query])
            
            # 4. 벡터 검색 (analysis와 image_summary만)
            search_results = self.vector_store.search(
                query_embedding[0], 
                top_k=Config.TOP_K_RETRIEVAL
            )
            
            # 5. 스마트 임계값으로 문서 연관성 판단
            smart_threshold = self.smart_threshold_calculator.calculate_smart_threshold(
                query, search_results, detected_domain
            )
            has_relevant_docs = self._check_document_relevance_enhanced(
                search_results, query, smart_threshold, keyword_result
            )
            
            if not search_results or not has_relevant_docs:
                # 문서와 연관 없는 질문 처리
                answer = self.answer_generator.generate_answer(
                    query, [], sources=[], has_relevant_docs=False
                )
                return {
                    "answer": answer,
                    "sources": [],
                    "confidence": 0.0,
                    "query_analysis": "",
                    "related_questions": []
                }
            
            # 4. 질문 의미 분석
            query_analysis = self._analyze_query_intent(query)
            
            # 5. 관련 질문들 검색 (questions 섹션에서)
            related_questions = self._get_related_questions(query_embedding[0], search_results)
            
            # 6. 리랭킹
            documents = [result["content"] for result in search_results]
            reranked_indices = self.reranker.rerank(query, documents, top_k=Config.TOP_K_RERANK)
            
            # 7. 최종 컨텍스트 구성
            context = []
            sources = []
            for idx, score in reranked_indices:
                result = search_results[idx]
                context.append(result["content"])
                sources.append({
                    "content": result["content"][:200] + "...",
                    "rerank_score": float(score),
                    "vector_score": float(result["score"]),
                    "metadata": result["metadata"]
                })
            
            # 8. 답변 생성 (출처 정보 포함)
            answer = self.answer_generator.generate_answer(
                query, context, sources=sources, has_relevant_docs=True
            )
            
            # 9. 적응형 가중치로 개선된 신뢰도 계산
            if reranked_indices:
                # 적응형 가중치 계산
                vector_weight, rerank_weight = self.adaptive_weight_calculator.calculate_adaptive_weights(
                    query, search_results, detected_domain
                )
                
                # 리랭킹 점수들을 정규화하여 0-1 범위로 변환
                rerank_scores = [abs(float(score)) for _, score in reranked_indices]
                vector_scores = [float(sources[i]["vector_score"]) for i in range(len(sources))]
                
                # 벡터 유사도와 리랭킹 점수를 결합
                if rerank_scores:
                    max_rerank = max(rerank_scores) if max(rerank_scores) > 0 else 1.0
                    normalized_rerank = [score / max_rerank for score in rerank_scores]
                    
                    # 적응형 가중 평균 계산
                    combined_scores = []
                    for i, (vec_score, rerank_score) in enumerate(zip(vector_scores, normalized_rerank)):
                        combined_score = (vec_score * vector_weight) + (rerank_score * rerank_weight)
                        combined_scores.append(combined_score)
                    
                    # 상위 결과들의 평균을 신뢰도로 사용
                    confidence = np.mean(combined_scores[:3])  # 상위 3개 평균
                    
                    logger.debug(f"적응형 신뢰도: {confidence:.3f} (벡터 {vector_weight:.2f}, 리랭킹 {rerank_weight:.2f})")
                else:
                    confidence = np.mean(vector_scores[:3])  # 벡터 점수만 사용
            else:
                confidence = 0.0
            
            # 신뢰도를 0-1 범위로 제한
            confidence = max(0.0, min(1.0, float(confidence)))
            
            # 10. 플라이휠 메트릭 수집
            result = {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "query_analysis": query_analysis,
                "related_questions": related_questions,
                "domain": detected_domain,
                "keywords": keyword_result['final_keywords'],
                "adaptive_weights": {
                    "vector": vector_weight if 'vector_weight' in locals() else 0.6,
                    "rerank": rerank_weight if 'rerank_weight' in locals() else 0.4
                },
                "smart_threshold": smart_threshold if 'smart_threshold' in locals() else 0.45
            }
            
            # 성능 메트릭 기록
            self.flywheel_metrics.record_query_performance(query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"검색 및 답변 생성 실패: {e}")
            return {
                "answer": "죄송합니다. 처리 중 오류가 발생했습니다.",
                "sources": [],
                "confidence": 0.0,
                "query_analysis": "",
                "related_questions": []
            }
    
    def _check_document_relevance(self, search_results: List[Dict], query: str = None) -> bool:
        """
        동적 임계값과 키워드 보정을 사용한 개선된 문서 연관성 판단
        """
        if not search_results:
            return False
        
        # 기본 점수 계산
        max_score = max(result["score"] for result in search_results)
        top_scores = [result["score"] for result in search_results[:3]]
        avg_score = np.mean(top_scores)
        
        # 동적 임계값 계산
        dynamic_threshold = self._calculate_dynamic_threshold(query, search_results)
        dynamic_avg_threshold = dynamic_threshold * 0.8
        
        # 키워드 보정 계산
        keyword_boost = self._calculate_keyword_boost(query, search_results)
        
        # 의미적 유사도 보정 계산
        semantic_boost = self._calculate_semantic_boost(query, search_results)
        
        # 종합 보정값 계산
        total_boost = keyword_boost + semantic_boost
        
        # 보정된 점수 계산
        adjusted_max_score = max_score + total_boost
        adjusted_avg_score = avg_score + total_boost
        
        # 1. 최고 점수 검증 (동적 임계값 사용)
        if adjusted_max_score < dynamic_threshold:
            logger.info(f"최고 점수 {adjusted_max_score:.3f} (키워드: +{keyword_boost:.3f}, 의미: +{semantic_boost:.3f}) < 동적 임계값 {dynamic_threshold:.3f}")
            return False
        
        # 2. 평균 점수 검증 (동적 임계값 사용)
        if adjusted_avg_score < dynamic_avg_threshold:
            logger.info(f"평균 점수 {adjusted_avg_score:.3f} (키워드: +{keyword_boost:.3f}, 의미: +{semantic_boost:.3f}) < 동적 평균 임계값 {dynamic_avg_threshold:.3f}")
            return False
        
        # 3. 추가 검증: 키워드 매칭률
        keyword_match_rate = self._calculate_keyword_match_rate(query, search_results)
        if keyword_match_rate < 0.1 and adjusted_max_score < 0.6:  # 키워드 매칭이 매우 낮고 점수도 보통인 경우
            logger.info(f"키워드 매칭률 {keyword_match_rate:.3f}이 너무 낮음 (점수: {adjusted_max_score:.3f})")
            return False
        
        logger.info(f"문서 연관성 확인됨 - 최고: {adjusted_max_score:.3f}, 평균: {adjusted_avg_score:.3f}, "
                   f"동적 임계값: {dynamic_threshold:.3f}, 키워드 매칭: {keyword_match_rate:.3f}, "
                   f"보정(키워드: +{keyword_boost:.3f}, 의미: +{semantic_boost:.3f})")
        return True
    
    def _check_document_relevance_enhanced(self, search_results: List[Dict], query: str, 
                                         smart_threshold: float, keyword_result: Dict) -> bool:
        """개선된 문서 연관성 판단 - 스마트 임계값 + 적응형 키워드 사용"""
        if not search_results:
            return False
        
        # 기본 점수 계산
        max_score = max(result["score"] for result in search_results)
        top_scores = [result["score"] for result in search_results[:3]]
        avg_score = np.mean(top_scores)
        
        # 적응형 키워드를 이용한 키워드 매칭 보정
        extracted_keywords = keyword_result.get('final_keywords', [])
        keyword_boost = self._calculate_keyword_boost_enhanced(extracted_keywords, search_results)
        
        # 의미적 유사도 보정 계산 (기존 로직 재사용)
        semantic_boost = self._calculate_semantic_boost(query, search_results)
        
        # 종합 보정값 계산
        total_boost = keyword_boost + semantic_boost
        
        # 보정된 점수 계산
        adjusted_max_score = max_score + total_boost
        adjusted_avg_score = avg_score + total_boost
        
        # 스마트 임계값 사용
        smart_avg_threshold = smart_threshold * 0.8
        
        # 1. 최고 점수 검증
        if adjusted_max_score < smart_threshold:
            logger.info(f"스마트 임계값 미달: 최고 점수 {adjusted_max_score:.3f} < {smart_threshold:.3f}")
            return False
        
        # 2. 평균 점수 검증
        if adjusted_avg_score < smart_avg_threshold:
            logger.info(f"스마트 평균 임계값 미달: 평균 점수 {adjusted_avg_score:.3f} < {smart_avg_threshold:.3f}")
            return False
        
        # 3. 키워드 매칭률 검증 (적응형 키워드 사용)
        keyword_match_rate = self._calculate_keyword_match_rate_enhanced(extracted_keywords, search_results)
        if keyword_match_rate < 0.1 and adjusted_max_score < 0.6:
            logger.info(f"키워드 매칭률 부족: {keyword_match_rate:.3f} (점수: {adjusted_max_score:.3f})")
            return False
        
        logger.info(f"스마트 문서 연관성 확인됨 - 최고: {adjusted_max_score:.3f}, 평균: {adjusted_avg_score:.3f}, "
                   f"스마트 임계값: {smart_threshold:.3f}, 키워드 매칭: {keyword_match_rate:.3f}, "
                   f"키워드 수: {len(extracted_keywords)}, 보정: +{total_boost:.3f}")
        return True
    
    def _calculate_keyword_boost_enhanced(self, keywords: List[str], search_results: List[Dict]) -> float:
        """적응형 키워드를 사용한 향상된 키워드 부스트 계산"""
        if not keywords or not search_results:
            return 0.0
        
        # 검색된 문서들에서 키워드 매칭 정도 계산
        total_matches = 0
        total_keywords = len(keywords)
        
        for result in search_results[:3]:  # 상위 3개 문서만 확인
            content = result.get("content", "").lower()
            matches = sum(1 for keyword in keywords if keyword.lower() in content)
            total_matches += matches
        
        # 매칭률 계산
        match_rate = total_matches / (total_keywords * 3) if total_keywords > 0 else 0.0
        
        # 보정값 계산 (최대 0.2까지 - 기존보다 강화)
        keyword_boost = min(0.2, match_rate * 0.4)
        
        logger.debug(f"향상된 키워드 보정: {keywords[:5]} → 매칭률 {match_rate:.3f} → 보정 +{keyword_boost:.3f}")
        
        return keyword_boost
    
    def _calculate_keyword_match_rate_enhanced(self, keywords: List[str], search_results: List[Dict]) -> float:
        """적응형 키워드를 사용한 향상된 키워드 매칭률 계산"""
        if not keywords or not search_results:
            return 0.0
        
        # 상위 문서들에서 키워드 매칭 확인
        matched_keywords = set()
        for result in search_results[:5]:
            content = result.get("content", "").lower()
            for keyword in keywords:
                if keyword.lower() in content:
                    matched_keywords.add(keyword)
        
        match_rate = len(matched_keywords) / len(keywords)
        return match_rate
    
    def _calculate_dynamic_threshold(self, query: str, search_results: List[Dict]) -> float:
        """
        질문과 검색 결과의 특성에 따라 동적 임계값을 계산합니다.
        """
        base_threshold = Config.SIMILARITY_THRESHOLD  # 0.45
        
        if not query:
            return base_threshold
        
        # 1. 질문 길이에 따른 조정
        query_length = len(query.strip())
        if query_length < 10:  # 매우 짧은 질문
            length_adjustment = 0.05  # 임계값 상승 (더 엄격)
        elif query_length > 50:  # 긴 질문
            length_adjustment = -0.03  # 임계값 하락 (더 관대)
        else:
            length_adjustment = 0.0
        
        # 2. 키워드 밀도에 따른 조정
        economic_keywords = self._count_economic_keywords(query)
        if economic_keywords >= 3:  # 경제 키워드가 많으면
            keyword_adjustment = -0.08  # 더 관대하게
        elif economic_keywords >= 1:
            keyword_adjustment = -0.04  # 약간 관대하게
        else:
            keyword_adjustment = 0.05   # 더 엄격하게
        
        # 3. 검색 결과 품질에 따른 조정
        if search_results:
            score_variance = np.var([result["score"] for result in search_results[:5]])
            if score_variance > 0.1:  # 점수 분산이 크면 (품질이 일관되지 않음)
                variance_adjustment = 0.03  # 더 엄격하게
            else:
                variance_adjustment = -0.02  # 더 관대하게
        else:
            variance_adjustment = 0.0
        
        # 4. 도메인별 조정
        domain_adjustment = self._get_domain_adjustment(query)
        
        # 최종 임계값 계산
        dynamic_threshold = base_threshold + length_adjustment + keyword_adjustment + variance_adjustment + domain_adjustment
        
        # 임계값 범위 제한 (0.25 ~ 0.65)
        dynamic_threshold = max(0.25, min(0.65, dynamic_threshold))
        
        logger.debug(f"동적 임계값 계산: 기본 {base_threshold:.3f} + 길이 {length_adjustment:.3f} + "
                    f"키워드 {keyword_adjustment:.3f} + 분산 {variance_adjustment:.3f} + "
                    f"도메인 {domain_adjustment:.3f} = {dynamic_threshold:.3f}")
        
        return dynamic_threshold
    
    def _calculate_keyword_boost(self, query: str, search_results: List[Dict]) -> float:
        """
        키워드 매칭 정도에 따른 점수 보정값을 계산합니다.
        """
        if not query or not search_results:
            return 0.0
        
        # 고급 키워드 추출 (기존 시스템 활용)
        try:
            if hasattr(self, 'answer_generator') and hasattr(self.answer_generator, '_extract_query_keywords'):
                keywords = self.answer_generator._extract_query_keywords(query)
            else:
                # 폴백: 기본 키워드 추출
                keywords = self._extract_basic_keywords(query)
        except:
            keywords = self._extract_basic_keywords(query)
        
        if not keywords:
            return 0.0
        
        # 검색된 문서들에서 키워드 매칭 정도 계산
        total_matches = 0
        total_keywords = len(keywords)
        
        for result in search_results[:3]:  # 상위 3개 문서만 확인
            content = result.get("content", "").lower()
            matches = sum(1 for keyword in keywords if keyword.lower() in content)
            total_matches += matches
        
        # 매칭률 계산
        match_rate = total_matches / (total_keywords * 3) if total_keywords > 0 else 0.0
        
        # 보정값 계산 (최대 0.15까지)
        keyword_boost = min(0.15, match_rate * 0.3)
        
        logger.debug(f"키워드 보정: {keywords} → 매칭률 {match_rate:.3f} → 보정 +{keyword_boost:.3f}")
        
        return keyword_boost
    
    def _calculate_keyword_match_rate(self, query: str, search_results: List[Dict]) -> float:
        """
        키워드 매칭률을 계산합니다.
        """
        if not query or not search_results:
            return 0.0
        
        try:
            if hasattr(self, 'answer_generator') and hasattr(self.answer_generator, '_extract_query_keywords'):
                keywords = self.answer_generator._extract_query_keywords(query)
            else:
                keywords = self._extract_basic_keywords(query)
        except:
            keywords = self._extract_basic_keywords(query)
        
        if not keywords:
            return 0.0
        
        # 상위 문서들에서 키워드 매칭 확인
        matched_keywords = set()
        for result in search_results[:5]:
            content = result.get("content", "").lower()
            for keyword in keywords:
                if keyword.lower() in content:
                    matched_keywords.add(keyword)
        
        match_rate = len(matched_keywords) / len(keywords)
        return match_rate
    
    def _count_economic_keywords(self, query: str) -> int:
        """경제/금융 관련 키워드 개수를 셉니다."""
        economic_terms = [
            "소비자물가", "cpi", "인플레이션", "고용", "실업", "일자리", "금리", "기준금리",
            "fomc", "연준", "통화정책", "양적완화", "qe", "금융시장", "주식", "채권",
            "경제", "성장", "gdp", "경기", "상승률", "하락", "증가", "감소"
        ]
        
        query_lower = query.lower()
        count = sum(1 for term in economic_terms if term in query_lower)
        return count
    
    def _get_domain_adjustment(self, query: str) -> float:
        """도메인별 임계값 조정값을 계산합니다."""
        query_lower = query.lower()
        
        # 핵심 경제 용어가 있으면 더 관대하게
        core_terms = ["소비자물가", "cpi", "고용지표", "실업률", "금리", "fomc"]
        if any(term in query_lower for term in core_terms):
            return -0.06
        
        # 일반 경제 용어가 있으면 약간 관대하게
        general_terms = ["경제", "시장", "정책", "지표", "상승", "하락"]
        if any(term in query_lower for term in general_terms):
            return -0.03
        
        # 경제와 무관한 용어가 있으면 더 엄격하게
        unrelated_terms = ["요리", "여행", "게임", "영화", "스포츠", "날씨"]
        if any(term in query_lower for term in unrelated_terms):
            return 0.10
        
        return 0.0
    
    def _extract_basic_keywords(self, query: str) -> List[str]:
        """기본적인 키워드 추출 (폴백용)"""
        basic_keywords = []
        economic_terms = [
            "소비자물가", "CPI", "인플레이션", "고용", "실업", "일자리", "금리", 
            "기준금리", "FOMC", "연준", "통화정책", "경제", "성장", "GDP"
        ]
        
        for term in economic_terms:
            if term in query or term.lower() in query.lower():
                basic_keywords.append(term)
        
        return basic_keywords
    
    def _calculate_semantic_boost(self, query: str, search_results: List[Dict]) -> float:
        """
        의미적 유사도를 기반으로 한 추가 보정값을 계산합니다.
        """
        if not query or not search_results:
            return 0.0
        
        try:
            # BGE-m3-ko 임베딩 모델 사용 (기존 시스템 활용)
            query_embedding = self.embedding_model.encode([query])
            
            # 검색된 문서들의 내용 추출
            doc_contents = [result.get("content", "")[:300] for result in search_results[:3]]  # 상위 3개, 300자 제한
            
            if not any(doc_contents):
                return 0.0
            
            # 문서 내용들의 임베딩 계산
            doc_embeddings = self.embedding_model.encode(doc_contents)
            
            # 코사인 유사도 계산
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # 의미적 유사도 분석
            max_similarity = max(similarities)
            avg_similarity = np.mean(similarities)
            
            # 경제/금융 도메인 특화 보정
            domain_boost = self._calculate_domain_semantic_boost(query, doc_contents)
            
            # 유사도 기반 보정값 계산
            # 높은 의미적 유사도일수록 더 큰 보정
            similarity_boost = 0.0
            
            if max_similarity > 0.8:  # 매우 높은 유사도
                similarity_boost = 0.10
            elif max_similarity > 0.6:  # 높은 유사도
                similarity_boost = 0.06
            elif max_similarity > 0.4:  # 중간 유사도
                similarity_boost = 0.03
            
            # 평균 유사도도 고려 (일관성 보너스)
            if avg_similarity > 0.5 and max_similarity > 0.6:
                similarity_boost += 0.02  # 일관성 보너스
            
            # 최종 의미적 보정값 (최대 0.12)
            semantic_boost = min(0.12, similarity_boost + domain_boost)
            
            logger.debug(f"의미적 보정: 최고 유사도 {max_similarity:.3f}, 평균 {avg_similarity:.3f}, "
                        f"도메인 보정 +{domain_boost:.3f} → 총 보정 +{semantic_boost:.3f}")
            
            return semantic_boost
            
        except Exception as e:
            logger.warning(f"의미적 유사도 보정 계산 실패: {e}")
            return 0.0
    
    def _calculate_domain_semantic_boost(self, query: str, doc_contents: List[str]) -> float:
        """
        경제/금융 도메인 특화 의미적 보정을 계산합니다.
        """
        # 경제/금융 도메인 핵심 개념들
        economic_concepts = {
            "inflation": ["인플레이션", "물가상승", "소비자물가", "CPI"],
            "employment": ["고용", "실업", "일자리", "취업", "노동시장"],
            "monetary_policy": ["금리", "통화정책", "FOMC", "연준", "기준금리"],
            "market": ["시장", "금융시장", "주식", "채권", "투자"],
            "growth": ["성장", "GDP", "경제성장", "경기", "회복"]
        }
        
        query_lower = query.lower()
        combined_docs = " ".join(doc_contents).lower()
        
        concept_matches = 0
        total_concepts = len(economic_concepts)
        
        for concept_name, keywords in economic_concepts.items():
            # 질문과 문서 모두에서 개념이 발견되면 매칭
            query_has_concept = any(keyword in query_lower for keyword in keywords)
            doc_has_concept = any(keyword in combined_docs for keyword in keywords)
            
            if query_has_concept and doc_has_concept:
                concept_matches += 1
        
        # 개념 매칭률에 따른 보정
        match_rate = concept_matches / total_concepts
        
        if match_rate >= 0.4:  # 40% 이상 매칭
            return 0.04
        elif match_rate >= 0.2:  # 20% 이상 매칭
            return 0.02
        else:
            return 0.0
    
    def _analyze_query_intent(self, query: str) -> str:
        """질문의 의도를 분석합니다."""
        query_lower = query.lower()
        
        # 키워드 기반 질문 분류
        if any(keyword in query for keyword in ["소비자물가", "cpi", "인플레이션"]):
            return "💰 소비자물가 관련 질문: 미국의 소비자물가지수(CPI) 동향, 인플레이션 압력, 물가 변화 등에 대해 문의하고 계십니다."
        
        elif any(keyword in query for keyword in ["고용", "실업", "일자리", "취업"]):
            return "👥 고용지표 관련 질문: 미국의 고용 상황, 실업률, 일자리 창출 등 노동시장 동향에 대해 문의하고 계십니다."
        
        elif any(keyword in query for keyword in ["금리", "기준금리", "fomc", "연준", "통화정책"]):
            return "🏦 통화정책 관련 질문: 미국 연방준비제도(Fed)의 금리 정책, FOMC 회의 결과, 통화정책 방향 등에 대해 문의하고 계십니다."
        
        elif any(keyword in query for keyword in ["금융시장", "주식", "채권", "시장반응"]):
            return "📈 금융시장 관련 질문: 경제지표 발표에 따른 금융시장 반응, 주식시장 및 채권시장 동향에 대해 문의하고 계십니다."
        
        elif any(keyword in query for keyword in ["경제", "성장", "gdp", "경기"]):
            return "🌍 경제전반 관련 질문: 미국의 경제 성장, GDP, 경기 전망 등 거시경제 상황에 대해 문의하고 계십니다."
        
        else:
            return "❓ 일반적인 경제 관련 질문: 한국은행이 분석한 미국 경제 관련 정보에 대해 문의하고 계십니다."
    
    def _get_related_questions(self, query_embedding: np.ndarray, search_results: List[Dict]) -> List[str]:
        """연관성 높은 질문들을 검색합니다."""
        try:
            # 가장 연관성 높은 문서의 파일명 찾기
            if not search_results:
                return []
            
            top_result = search_results[0]
            file_name = top_result["metadata"].get("file_name", "")
            
            # questions 섹션 검색
            question_results = self.vector_store.search_questions(query_embedding, top_k=3)
            
            # 같은 파일의 질문들 우선 선택
            related_questions = []
            
            # 1. 같은 파일의 질문들
            for result in question_results:
                if result["metadata"].get("file_name") == file_name:
                    questions_text = result["content"]
                    # 개별 질문으로 분리
                    questions = self._parse_questions(questions_text)
                    related_questions.extend(questions[:3])  # 최대 3개
            
            # 2. 다른 파일의 질문들 (같은 파일 질문이 부족한 경우)
            if len(related_questions) < 3:
                for result in question_results:
                    if result["metadata"].get("file_name") != file_name:
                        questions_text = result["content"]
                        questions = self._parse_questions(questions_text)
                        remaining_slots = 3 - len(related_questions)
                        related_questions.extend(questions[:remaining_slots])
                        if len(related_questions) >= 3:
                            break
            
            return related_questions[:3]  # 최대 3개 반환
            
        except Exception as e:
            logger.error(f"관련 질문 검색 실패: {e}")
            return []
    
    def _parse_questions(self, questions_text: str) -> List[str]:
        """질문 텍스트에서 개별 질문들을 파싱합니다."""
        questions = []
        lines = questions_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # 숫자로 시작하는 질문 찾기 (예: "1. 질문내용?")
            if line and (line[0].isdigit() or line.startswith('-')):
                # 숫자와 점 제거
                question = line.split('.', 1)[-1].strip()
                if question and question.endswith('?'):
                    questions.append(question)
        
        return questions 