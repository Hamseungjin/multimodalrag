"""
Gemma-3-12b-it 기반 로컬 합성 Q&A 생성기
API 없이 완전 로컬에서 동작하는 플라이휠 워크플로우용 데이터 생성
"""
import torch
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from config import Config

class LocalSyntheticQAGenerator:
    """Gemma-3-12b-it 기반 로컬 합성 Q&A 생성기"""
    
    def __init__(self, model_name: str = Config.LLM_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 도메인별 프롬프트 템플릿
        self.domain_prompts = {
            "economics": {
                "system": "당신은 경제 전문가입니다. 한국은행 뉴스와 경제 지표를 바탕으로 현실적이고 구체적인 경제 관련 질문과 답변을 생성해주세요.",
                "topics": [
                    "소비자물가지수(CPI)", "고용률", "실업률", "GDP 성장률", "기준금리", 
                    "인플레이션", "디플레이션", "통화정책", "재정정책", "경제성장"
                ],
                "question_types": [
                    "최근 동향 질문", "원인 분석 질문", "전망 질문", "비교 분석 질문", "정책 영향 질문"
                ]
            },
            "finance": {
                "system": "당신은 금융 전문가입니다. 금융시장, 투자, 자산관리에 대한 실용적인 질문과 답변을 생성해주세요.",
                "topics": [
                    "주식투자", "채권투자", "포트폴리오", "리스크관리", "자산배분",
                    "금융상품", "투자전략", "시장분석", "환율", "금리"
                ],
                "question_types": [
                    "투자 조언 질문", "위험도 분석 질문", "상품 비교 질문", "시장 전망 질문", "전략 수립 질문"
                ]
            },
            "healthcare": {
                "system": "당신은 의료 전문가입니다. 의학 정보, 건강관리, 질병에 대한 정확하고 도움이 되는 질문과 답변을 생성해주세요.",
                "topics": [
                    "질병 예방", "건강검진", "만성질환", "응급처치", "영양관리",
                    "운동요법", "스트레스 관리", "수면건강", "정신건강", "노인건강"
                ],
                "question_types": [
                    "증상 관련 질문", "예방법 질문", "치료법 질문", "생활습관 질문", "건강관리 질문"
                ]
            },
            "legal": {
                "system": "당신은 법률 전문가입니다. 법률 정보, 권리구제, 법적 절차에 대한 명확하고 실용적인 질문과 답변을 생성해주세요.",
                "topics": [
                    "민법", "형법", "상법", "노동법", "부동산법",
                    "계약법", "소송절차", "법률상담", "권리구제", "법적책임"
                ],
                "question_types": [
                    "법적 권리 질문", "절차 안내 질문", "계약 관련 질문", "분쟁해결 질문", "법적 의무 질문"
                ]
            }
        }
        
        self.load_model()
    
    def load_model(self):
        """Gemma 모델 로드"""
        try:
            logger.info(f"Gemma 모델 로딩 중: {self.model_name}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 로드 (메모리 최적화)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("✅ Gemma 모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"❌ Gemma 모델 로딩 실패: {e}")
            raise
    
    def generate_qa_pairs_by_domain(self, domain: str, count: int = 10, 
                                   context_data: List[str] = None) -> List[Dict[str, Any]]:
        """도메인별 합성 Q&A 쌍 생성"""
        if domain not in self.domain_prompts:
            raise ValueError(f"지원하지 않는 도메인: {domain}")
        
        logger.info(f"📝 {domain} 도메인 Q&A {count}개 생성 시작")
        
        domain_config = self.domain_prompts[domain]
        qa_pairs = []
        
        for i in range(count):
            try:
                # 랜덤하게 주제와 질문 유형 선택
                import random
                topic = random.choice(domain_config["topics"])
                question_type = random.choice(domain_config["question_types"])
                
                # 프롬프트 생성
                prompt = self._create_qa_generation_prompt(domain, topic, question_type, context_data)
                
                # Q&A 생성
                generated_text = self._generate_text(prompt)
                
                # 결과 파싱
                qa_pair = self._parse_qa_from_text(generated_text, domain, topic, question_type)
                
                if qa_pair:
                    qa_pairs.append(qa_pair)
                    logger.debug(f"생성 완료 {i+1}/{count}: {qa_pair['question'][:50]}...")
                else:
                    logger.warning(f"Q&A 파싱 실패 {i+1}/{count}")
                
            except Exception as e:
                logger.error(f"Q&A 생성 실패 {i+1}/{count}: {e}")
                continue
        
        logger.info(f"✅ {domain} 도메인 Q&A 생성 완료: {len(qa_pairs)}개")
        return qa_pairs
    
    def _create_qa_generation_prompt(self, domain: str, topic: str, question_type: str, 
                                   context_data: List[str] = None) -> str:
        """도메인별 Q&A 생성 프롬프트 작성"""
        domain_config = self.domain_prompts[domain]
        
        # 기본 프롬프트
        prompt = f"""<start_of_turn>user
{domain_config['system']}

주제: {topic}
질문 유형: {question_type}

다음 형식으로 1개의 질문과 답변을 생성해주세요:

**질문:** [구체적이고 실용적인 질문]
**답변:** [정확하고 도움이 되는 답변, 2-3문장]

"""
        
        # 컨텍스트 데이터가 있으면 추가
        if context_data and len(context_data) > 0:
            sample_context = context_data[0][:500]  # 500자로 제한
            prompt += f"""
참고 정보:
{sample_context}

위 정보를 참고하여 질문과 답변을 생성해주세요.
"""
        
        prompt += "<end_of_turn>\n<start_of_turn>model\n"
        
        return prompt
    
    def _generate_text(self, prompt: str, max_length: int = 512) -> str:
        """텍스트 생성"""
        try:
            # 토큰화
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # 생성 설정
            generation_config = {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
            
            # 생성 실행
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_config
                )
            
            # 결과 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"텍스트 생성 실패: {e}")
            return ""
    
    def _parse_qa_from_text(self, generated_text: str, domain: str, topic: str, 
                           question_type: str) -> Optional[Dict[str, Any]]:
        """생성된 텍스트에서 Q&A 파싱"""
        try:
            # 질문과 답변 추출 패턴
            question_patterns = [
                r"\*\*질문:\*\*\s*(.+?)(?=\*\*답변:\*\*|\n\n|\Z)",
                r"질문:\s*(.+?)(?=답변:|\n\n|\Z)",
                r"Q:\s*(.+?)(?=A:|\n\n|\Z)"
            ]
            
            answer_patterns = [
                r"\*\*답변:\*\*\s*(.+?)(?=\n\n|\Z)",
                r"답변:\s*(.+?)(?=\n\n|\Z)",
                r"A:\s*(.+?)(?=\n\n|\Z)"
            ]
            
            question = None
            answer = None
            
            # 질문 추출
            for pattern in question_patterns:
                match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
                if match:
                    question = match.group(1).strip()
                    break
            
            # 답변 추출
            for pattern in answer_patterns:
                match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    break
            
            # 품질 검증
            if question and answer and len(question) > 10 and len(answer) > 20:
                qa_pair = {
                    "question": question,
                    "answer": answer,
                    "domain": domain,
                    "topic": topic,
                    "question_type": question_type,
                    "generated_at": time.time(),
                    "quality_score": self._calculate_qa_quality(question, answer),
                    "source": "local_gemma"
                }
                
                # 품질 임계값 확인
                if qa_pair["quality_score"] >= Config.QUALITY_THRESHOLD:
                    return qa_pair
            
            return None
            
        except Exception as e:
            logger.error(f"Q&A 파싱 실패: {e}")
            return None
    
    def _calculate_qa_quality(self, question: str, answer: str) -> float:
        """Q&A 품질 점수 계산"""
        quality_score = 0.0
        
        # 1. 길이 기반 점수 (30%)
        question_len = len(question)
        answer_len = len(answer)
        
        if 15 <= question_len <= 100:
            quality_score += 0.15
        if 30 <= answer_len <= 300:
            quality_score += 0.15
        
        # 2. 구조 기반 점수 (25%)
        if question.endswith('?') or '어떻' in question or '무엇' in question:
            quality_score += 0.1
        if '. ' in answer or '다.' in answer:  # 문장 구조
            quality_score += 0.15
        
        # 3. 컨텐츠 기반 점수 (25%)
        question_words = len(question.split())
        answer_words = len(answer.split())
        
        if 3 <= question_words <= 20:
            quality_score += 0.1
        if 10 <= answer_words <= 50:
            quality_score += 0.15
        
        # 4. 반복/노이즈 검사 (20%)
        if not self._has_repetition(question) and not self._has_repetition(answer):
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def _has_repetition(self, text: str) -> bool:
        """텍스트 내 반복 패턴 검사"""
        words = text.split()
        if len(words) < 4:
            return False
        
        # 연속 단어 반복 검사
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
        
        return False
    
    def batch_generate_qa_pairs(self, domains: List[str], count_per_domain: int = 20,
                               save_path: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
        """여러 도메인에 대해 배치로 Q&A 생성"""
        all_qa_pairs = {}
        
        for domain in domains:
            logger.info(f"🎯 {domain} 도메인 배치 생성 시작")
            
            qa_pairs = self.generate_qa_pairs_by_domain(domain, count_per_domain)
            all_qa_pairs[domain] = qa_pairs
            
            logger.info(f"✅ {domain}: {len(qa_pairs)}개 생성 완료")
        
        # 결과 저장
        if save_path:
            self._save_qa_pairs(all_qa_pairs, save_path)
        
        # 통계 출력
        total_count = sum(len(pairs) for pairs in all_qa_pairs.values())
        avg_quality = sum(
            sum(pair["quality_score"] for pair in pairs) / len(pairs)
            for pairs in all_qa_pairs.values() if pairs
        ) / len(all_qa_pairs)
        
        logger.info(f"🎊 배치 생성 완료: 총 {total_count}개, 평균 품질 {avg_quality:.3f}")
        
        return all_qa_pairs
    
    def _save_qa_pairs(self, qa_pairs_dict: Dict[str, List[Dict[str, Any]]], save_path: Path):
        """Q&A 쌍을 파일로 저장"""
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Q&A 데이터 저장: {save_path}")
            
        except Exception as e:
            logger.error(f"Q&A 저장 실패: {e}")
    
    def cleanup_model(self):
        """모델 메모리 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🧹 모델 메모리 정리 완료")

# 사용 예시 함수들
def generate_economics_qa_sample():
    """경제 도메인 Q&A 생성 예시"""
    generator = LocalSyntheticQAGenerator()
    
    try:
        qa_pairs = generator.generate_qa_pairs_by_domain("economics", count=5)
        
        print("📊 생성된 경제 Q&A 샘플:")
        print("=" * 50)
        
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\n{i}. 주제: {qa['topic']} | 유형: {qa['question_type']}")
            print(f"질문: {qa['question']}")
            print(f"답변: {qa['answer']}")
            print(f"품질: {qa['quality_score']:.3f}")
            print("-" * 40)
        
        return qa_pairs
        
    finally:
        generator.cleanup_model()

def generate_multi_domain_qa():
    """다중 도메인 Q&A 생성 예시"""
    generator = LocalSyntheticQAGenerator()
    
    try:
        domains = ["economics", "healthcare", "legal"]
        qa_results = generator.batch_generate_qa_pairs(
            domains, 
            count_per_domain=3,
            save_path=Path("generated_qa_multi_domain.json")
        )
        
        # 도메인별 결과 출력
        for domain, qa_pairs in qa_results.items():
            print(f"\n🎯 {domain.upper()} 도메인 ({len(qa_pairs)}개):")
            for qa in qa_pairs[:2]:  # 각 도메인당 2개씩만 출력
                print(f"Q: {qa['question'][:60]}...")
                print(f"A: {qa['answer'][:80]}...")
                print()
        
        return qa_results
        
    finally:
        generator.cleanup_model()

if __name__ == "__main__":
    # 로깅 설정
    logger.add("synthetic_qa_generation.log", rotation="10 MB")
    
    print("🚀 Gemma-3-12b-it 기반 로컬 합성 Q&A 생성기")
    print("1. 경제 도메인 샘플 생성")
    # generate_economics_qa_sample()
    
    print("\n2. 다중 도메인 배치 생성")
    # generate_multi_domain_qa()
    
    print("\n✅ 테스트 완료! 실제 사용을 위해서는 주석을 해제하세요.") 