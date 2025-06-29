"""
적응형 RAG 컴포넌트들 - 플라이휠 워크플로우 지원
개선사항:
1. 적응형 키워드 추출 + 캐싱
2. 동적 가중치 계산 (벡터/리랭킹)
3. 스마트 임계값 계산
4. 플라이휠 메트릭 수집
"""
import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import re
import random
import uuid
from kiwipiepy import Kiwi
from common_constants import (
    DOMAIN_SEED_KEYWORDS, SUPPORTED_DOMAINS, MAX_CACHE_SIZE, 
    CACHE_EXPIRY_SECONDS, ECONOMIC_TERMS, OPTIMAL_WEIGHTS
)

# 전역 캐시 변수들
KEYWORD_CACHE = {}
WEIGHT_CACHE = {}
THRESHOLD_CACHE = {}
DOMAIN_DETECTION_CACHE = {}  # LLM 도메인 감지 결과 캐시

class DomainDetector:
    """도메인 자동 감지 클래스 - 키워드 매칭 + LLM 기반 하이브리드"""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model
        self.use_llm_detection = llm_model is not None
        
        # LLM 도메인 감지용 프롬프트
        self.domain_detection_prompt = """다음 텍스트의 도메인을 정확히 분류해주세요.

텍스트: "{text}"

도메인 옵션:
- economics: 경제, 금융정책, 물가, 고용, GDP 등 거시경제
- finance: 투자, 주식, 채권, 자산관리, 금융상품 등
- healthcare: 의료, 건강, 질병, 치료, 병원 등
- technology: IT기술, 소프트웨어, AI, 컴퓨터 등
- legal: 법률, 계약, 소송, 권리의무 등
- general: 위 카테고리에 해당하지 않는 일반적인 내용

응답 형식: 도메인명만 정확히 반환 (예: economics)"""
    
    def detect_domain(self, text: str, use_llm: bool = True) -> str:
        """텍스트에서 도메인을 자동 감지 (하이브리드 방식)"""
        
        # 1. 캐시 확인
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in DOMAIN_DETECTION_CACHE:
            cached_data = DOMAIN_DETECTION_CACHE[cache_key]
            if time.time() - cached_data['timestamp'] < 7200:  # 2시간 캐시
                logger.debug(f"도메인 캐시 히트: {cached_data['domain']}")
                return cached_data['domain']
        
        # 2. LLM 기반 감지 시도 (우선순위)
        if self.use_llm_detection and use_llm:
            try:
                llm_domain = self._detect_domain_with_llm(text)
                if llm_domain and llm_domain in SUPPORTED_DOMAINS:
                    self._save_domain_to_cache(cache_key, llm_domain, method="llm")
                    return llm_domain
                else:
                    logger.warning(f"LLM 도메인 감지 결과 무효: {llm_domain}")
            except Exception as e:
                logger.warning(f"LLM 도메인 감지 실패, 키워드 방식으로 fallback: {e}")
        
        # 3. 키워드 기반 감지 (fallback)
        keyword_domain = self._detect_domain_with_keywords(text)
        self._save_domain_to_cache(cache_key, keyword_domain, method="keyword")
        
        return keyword_domain
    
    def _detect_domain_with_llm(self, text: str) -> Optional[str]:
        """LLM을 사용한 도메인 감지"""
        if not self.llm_model:
            return None
        
        try:
            prompt = self.domain_detection_prompt.format(text=text[:500])  # 길이 제한
            
            # LLM 추론 실행
            response = self._generate_llm_response(prompt)
            
            # 응답에서 도메인 추출
            domain = self._parse_domain_response(response)
            
            logger.debug(f"LLM 도메인 감지: '{text[:50]}...' → {domain}")
            return domain
            
        except Exception as e:
            logger.error(f"LLM 도메인 감지 중 오류: {e}")
            return None
    
    def _generate_llm_response(self, prompt: str) -> str:
        """LLM 응답 생성"""
        try:
            # Gemma 모델 사용 (이미 로드된 경우)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            if hasattr(self.llm_model, 'generate'):
                # 이미 로드된 모델 사용
                tokenizer = getattr(self.llm_model, 'tokenizer', None)
                if not tokenizer:
                    # 토크나이저가 없으면 새로 로드
                    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
                
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        inputs.input_ids,
                        max_new_tokens=50,
                        temperature=0.1,  # 낮은 온도로 일관성 확보
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return response.strip()
            else:
                # 모델이 로드되지 않은 경우 키워드 방식으로 fallback
                logger.warning("LLM 모델이 로드되지 않음, 키워드 방식 사용")
                return ""
                
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            return ""
    
    def _parse_domain_response(self, response: str) -> Optional[str]:
        """LLM 응답에서 도메인 추출"""
        response = response.lower().strip()
        
        # 직접 도메인명이 포함된 경우
        for domain in SUPPORTED_DOMAINS:
            if domain in response:
                return domain
        
        # 한국어 키워드 매핑
        korean_domain_map = {
            "경제": "economics",
            "금융": "finance", 
            "의료": "healthcare",
            "건강": "healthcare",
            "기술": "technology",
            "법률": "legal",
            "일반": "general"
        }
        
        for korean, english in korean_domain_map.items():
            if korean in response:
                return english
        
        return None
    
    def _detect_domain_with_keywords(self, text: str) -> str:
        """키워드 기반 도메인 감지 (단순 fallback)"""
        text_lower = text.lower()
        
        # 간단한 키워드 체크
        if any(word in text_lower for word in ["경제", "물가", "인플레이션", "GDP", "고용", "실업"]):
            return "economics"
        elif any(word in text_lower for word in ["투자", "주식", "채권", "자산", "수익률"]):
            return "finance" 
        elif any(word in text_lower for word in ["의료", "건강", "질병", "치료", "병원"]):
            return "healthcare"
        elif any(word in text_lower for word in ["기술", "소프트웨어", "AI", "컴퓨터"]):
            return "technology"
        elif any(word in text_lower for word in ["법률", "법", "판결", "소송", "변호사"]):
            return "legal"
        else:
            return "general"
    
    def _save_domain_to_cache(self, cache_key: str, domain: str, method: str):
        """도메인 감지 결과를 캐시에 저장"""
        global DOMAIN_DETECTION_CACHE
        
        # 캐시 크기 관리
        if len(DOMAIN_DETECTION_CACHE) >= MAX_CACHE_SIZE:
            oldest_items = sorted(DOMAIN_DETECTION_CACHE.items(), 
                                key=lambda x: x[1]['timestamp'])
            for i in range(len(oldest_items) // 4):
                del DOMAIN_DETECTION_CACHE[oldest_items[i][0]]
        
        DOMAIN_DETECTION_CACHE[cache_key] = {
            'domain': domain,
            'method': method,
            'timestamp': time.time()
        }
        
        logger.debug(f"도메인 캐시 저장: {domain} (method: {method})")

class AdaptiveKeywordExtractor:
    """적응형 키워드 추출기 - Kiwi 형태소 분석기 기반"""
    
    def __init__(self, embedding_model, llm_model=None):
        self.embedding_model = embedding_model
        self.domain_detector = DomainDetector(llm_model)
        self.tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
        
        # Kiwi 형태소 분석기 초기화
        try:
            self.kiwi = Kiwi()
            self.use_kiwi = True
            logger.info("✅ Kiwi 형태소 분석기 로드 완료")
        except Exception as e:
            logger.warning(f"Kiwi 로드 실패, 정규식 방식 사용: {e}")
            self.kiwi = None
            self.use_kiwi = False
        
        # 도메인별 시드 키워드 (공통 상수 사용)
        self.domain_seed_keywords = DOMAIN_SEED_KEYWORDS
        
        self.domain_embeddings_cache = {}
        self._initialize_domain_embeddings()
    
    def _initialize_domain_embeddings(self):
        """도메인별 시드 키워드 임베딩을 미리 계산"""
        for domain, categories in self.domain_seed_keywords.items():
            if domain not in self.domain_embeddings_cache:
                self.domain_embeddings_cache[domain] = {}
            
            for category, keywords in categories.items():
                try:
                    embeddings = self.embedding_model.encode(keywords)
                    self.domain_embeddings_cache[domain][category] = embeddings
                except Exception as e:
                    logger.warning(f"도메인 {domain} 임베딩 생성 실패: {e}")
    
    def extract_keywords_adaptive(self, query: str, context_texts: List[str] = None, 
                                 domain: str = None) -> Dict[str, Any]:
        """적응형 키워드 추출"""
        detected_domain = domain or self.domain_detector.detect_domain(query)
        
        # 캐시 확인
        cache_key = hashlib.md5(f"{detected_domain}:{query}".encode()).hexdigest()
        if cache_key in KEYWORD_CACHE:
            cached_data = KEYWORD_CACHE[cache_key]
            if time.time() - cached_data['timestamp'] < 3600:  # 1시간 캐시
                return cached_data['keywords']
        
        # 새로운 키워드 추출
        result = self._extract_fresh_keywords(query, context_texts, detected_domain)
        
        # 캐시에 저장
        if len(KEYWORD_CACHE) >= MAX_CACHE_SIZE:
            # 오래된 캐시 제거
            oldest_items = sorted(KEYWORD_CACHE.items(), 
                                key=lambda x: x[1]['timestamp'])
            for i in range(len(oldest_items) // 4):
                del KEYWORD_CACHE[oldest_items[i][0]]
        
        KEYWORD_CACHE[cache_key] = {
            'keywords': result,
            'timestamp': time.time()
        }
        
        return result
    
    def _extract_fresh_keywords(self, query: str, context_texts: List[str], 
                               domain: str) -> Dict[str, Any]:
        """새로운 키워드 추출"""
        result = {
            "domain": domain,
            "basic_keywords": [],
            "semantic_keywords": [],
            "tfidf_keywords": [],
            "final_keywords": []
        }
        
        try:
            # 1. 기본 키워드 추출
            basic_keywords = self._extract_basic_keywords(query)
            result["basic_keywords"] = basic_keywords
            
            # 2. 도메인별 의미적 확장
            semantic_keywords = self._expand_keywords_by_domain(query, basic_keywords, domain)
            result["semantic_keywords"] = semantic_keywords
            
            # 3. TF-IDF 키워드
            if context_texts:
                tfidf_keywords = self._extract_tfidf_keywords(query, context_texts)
                result["tfidf_keywords"] = tfidf_keywords
            
            # 4. 최종 통합
            final_keywords = self._integrate_keywords_adaptive(
                basic_keywords, semantic_keywords, result["tfidf_keywords"], domain
            )
            result["final_keywords"] = final_keywords
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            result["final_keywords"] = self._extract_basic_keywords(query)
        
        return result
    
    def _extract_basic_keywords(self, query: str) -> List[str]:
        """Kiwi 형태소 분석기를 사용한 정밀 키워드 추출"""
        keywords = []
        
        if self.use_kiwi and self.kiwi:
            try:
                # Kiwi 형태소 분석
                tokens = self.kiwi.analyze(query)
                
                for token_info in tokens[0][0]:  # 첫 번째 분석 결과 사용
                    morph = token_info[0]  # 형태소
                    pos = token_info[1]    # 품사
                    
                    # 명사, 고유명사, 영어, 한자어 추출
                    if pos in ['NNG', 'NNP', 'SL', 'SH'] and len(morph) >= 2:
                        keywords.append(morph)
                    
                    # 복합명사 처리 (연속된 명사들 결합)
                    elif pos in ['NNG', 'NNP'] and len(morph) >= 1:
                        keywords.append(morph)
                
                # 중복 제거 및 품질 필터링
                keywords = self._filter_quality_keywords(list(set(keywords)))
                
                logger.debug(f"Kiwi 키워드 추출: {keywords}")
                
            except Exception as e:
                logger.warning(f"Kiwi 분석 실패, 정규식 방식으로 fallback: {e}")
                keywords = self._extract_keywords_regex(query)
        else:
            # Kiwi 사용 불가시 정규식 방식
            keywords = self._extract_keywords_regex(query)
        
        return keywords
    
    def _extract_keywords_regex(self, query: str) -> List[str]:
        """정규식 기반 키워드 추출 (fallback)"""
        keywords = []
        
        # 한글 키워드 (2글자 이상)
        korean_pattern = r'[가-힣]{2,}'
        korean_words = re.findall(korean_pattern, query)
        keywords.extend(korean_words)
        
        # 영어 대문자 약어
        english_pattern = r'[A-Z]{2,}'
        english_words = re.findall(english_pattern, query)
        keywords.extend(english_words)
        
        # 숫자+한글 조합 (예: "3분기", "2023년")
        number_korean_pattern = r'\d+[가-힣]+'
        number_korean = re.findall(number_korean_pattern, query)
        keywords.extend(number_korean)
        
        return list(set(keywords))
    
    def _filter_quality_keywords(self, keywords: List[str]) -> List[str]:
        """키워드 품질 필터링"""
        filtered = []
        
        # 불용어 제거
        stopwords = {'이다', '있다', '하다', '되다', '것', '수', '등', '및', '또는', '그리고', '하지만', '때문', '위해'}
        
        for keyword in keywords:
            # 길이 조건 (2-15자)
            if not (2 <= len(keyword) <= 15):
                continue
            
            # 불용어 제거
            if keyword in stopwords:
                continue
                
            # 단순 반복 문자 제거 (예: "ㅋㅋㅋ", "...")
            if len(set(keyword)) <= 2 and len(keyword) > 3:
                continue
            
            # 숫자만으로 구성된 경우 제외 (단, 년도는 포함)
            if keyword.isdigit() and not (1900 <= int(keyword) <= 2100):
                continue
            
            filtered.append(keyword)
        
        return filtered
    
    def _expand_keywords_by_domain(self, query: str, base_keywords: List[str], 
                                  domain: str) -> List[str]:
        """도메인별 의미적 키워드 확장"""
        if not base_keywords or domain not in self.domain_embeddings_cache:
            return []
        
        expanded_keywords = []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            domain_embeddings = self.domain_embeddings_cache[domain]
            
            for category, seed_embeddings in domain_embeddings.items():
                similarities = cosine_similarity(query_embedding, seed_embeddings)[0]
                threshold = 0.5 if domain == "economics" else 0.55
                
                for i, similarity in enumerate(similarities):
                    if similarity > threshold:
                        seed_keywords = self.domain_seed_keywords[domain][category]
                        keyword = seed_keywords[i]
                        if keyword not in base_keywords and keyword not in expanded_keywords:
                            expanded_keywords.append(keyword)
            
        except Exception as e:
            logger.warning(f"도메인별 키워드 확장 실패: {e}")
        
        return expanded_keywords[:8]
    
    def _extract_tfidf_keywords(self, query: str, context_texts: List[str]) -> List[str]:
        """TF-IDF 기반 키워드 추출"""
        if not context_texts:
            return []
        
        try:
            all_texts = [query] + context_texts
            tfidf_matrix = self.tfidf.fit_transform(all_texts)
            feature_names = self.tfidf.get_feature_names_out()
            query_tfidf = tfidf_matrix[0].toarray()[0]
            
            word_scores = [(feature_names[i], score) 
                          for i, score in enumerate(query_tfidf) if score > 0]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            tfidf_keywords = []
            for word, score in word_scores[:15]:
                if any(ord('가') <= ord(char) <= ord('힣') for char in word) or word.isupper():
                    tfidf_keywords.append(word)
            
            return tfidf_keywords[:6]
            
        except Exception as e:
            logger.warning(f"TF-IDF 키워드 추출 실패: {e}")
            return []
    
    def _integrate_keywords_adaptive(self, basic_keywords: List[str], semantic_keywords: List[str], 
                                   tfidf_keywords: List[str], domain: str) -> List[str]:
        """도메인별 적응형 키워드 통합"""
        keyword_scores = {}
        
        # 도메인별 가중치
        if domain == "economics":
            basic_weight, semantic_weight, tfidf_weight = 1.0, 0.9, 0.7
        elif domain == "finance":
            basic_weight, semantic_weight, tfidf_weight = 1.0, 0.8, 0.8
        else:
            basic_weight, semantic_weight, tfidf_weight = 1.0, 0.6, 0.5
        
        for keyword in basic_keywords:
            keyword_scores[keyword] = keyword_scores.get(keyword, 0) + basic_weight
        
        for keyword in semantic_keywords:
            keyword_scores[keyword] = keyword_scores.get(keyword, 0) + semantic_weight
        
        for keyword in tfidf_keywords:
            keyword_scores[keyword] = keyword_scores.get(keyword, 0) + tfidf_weight
        
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        min_score = 0.5 if domain == "economics" else 0.4
        
        return [keyword for keyword, score in sorted_keywords if score >= min_score]

class AdaptiveWeightCalculator:
    """적응형 가중치 계산기 - 벡터/리랭킹 가중치 동적 조정"""
    
    def __init__(self):
        self.optimal_weights = OPTIMAL_WEIGHTS
    
    def calculate_adaptive_weights(self, query: str, search_results: List[Dict], 
                                 domain: str = "general") -> Tuple[float, float]:
        """적응형 가중치 계산"""
        # 캐시 확인
        cache_key = f"{domain}:{len(search_results)}:{len(query)}"
        if cache_key in WEIGHT_CACHE:
            cached_weights = WEIGHT_CACHE[cache_key]
            if time.time() - cached_weights['timestamp'] < 1800:  # 30분 캐시
                return cached_weights['vector'], cached_weights['rerank']
        
        # 질문 특성 분석
        query_features = self._analyze_query_features(query, search_results)
        
        # 도메인별 기본 가중치
        base_weights = self.optimal_weights.get(domain, self.optimal_weights["general"])
        vector_weight = base_weights["vector"]
        rerank_weight = base_weights["rerank"]
        
        # 질문 유형별 조정
        if query_features["is_factual"]:
            vector_weight += 0.1
            rerank_weight -= 0.1
        elif query_features["is_analytical"]:
            vector_weight -= 0.05
            rerank_weight += 0.05
        
        # 검색 결과 품질에 따른 조정
        if query_features["result_quality"] > 0.8:
            vector_weight += 0.05
        elif query_features["result_quality"] < 0.5:
            rerank_weight += 0.1
            vector_weight -= 0.1
        
        # 정규화
        total = vector_weight + rerank_weight
        vector_weight /= total
        rerank_weight /= total
        
        # 캐시에 저장
        WEIGHT_CACHE[cache_key] = {
            'vector': vector_weight,
            'rerank': rerank_weight,
            'timestamp': time.time()
        }
        
        logger.debug(f"적응형 가중치: 벡터 {vector_weight:.3f}, 리랭킹 {rerank_weight:.3f}")
        
        return vector_weight, rerank_weight
    
    def _analyze_query_features(self, query: str, search_results: List[Dict]) -> Dict[str, Any]:
        """쿼리와 검색 결과의 특성 분석"""
        features = {}
        
        # 질문 유형 분석
        factual_indicators = ["무엇", "언제", "어디", "누구", "얼마", "몇"]
        analytical_indicators = ["왜", "어떻게", "분석", "비교", "평가", "전망"]
        
        features["is_factual"] = any(indicator in query for indicator in factual_indicators)
        features["is_analytical"] = any(indicator in query for indicator in analytical_indicators)
        
        # 검색 결과 품질 평가
        if search_results:
            scores = [result.get("score", 0) for result in search_results[:5]]
            features["result_quality"] = np.mean(scores)
            features["score_variance"] = np.var(scores)
        else:
            features["result_quality"] = 0.0
            features["score_variance"] = 0.0
        
        return features

class SmartThresholdCalculator:
    """스마트 임계값 계산기"""
    
    def calculate_smart_threshold(self, query: str, search_results: List[Dict], 
                                domain: str = "general") -> float:
        """ML 기반 스마트 임계값 계산"""
        from config import Config
        base_threshold = Config.SIMILARITY_THRESHOLD
        
        # 특성 벡터 생성
        features = self._extract_threshold_features(query, search_results, domain)
        
        # 규칙 기반 임계값 계산
        smart_threshold = self._rule_based_threshold(features, base_threshold)
        
        return smart_threshold
    
    def _extract_threshold_features(self, query: str, search_results: List[Dict], 
                                  domain: str) -> Dict[str, float]:
        """임계값 결정을 위한 특성 추출"""
        features = {}
        
        # 쿼리 특성
        features["query_length"] = len(query) / 100.0
        features["query_complexity"] = len(query.split()) / 20.0
        features["has_economic_terms"] = self._count_economic_terms(query) / 10.0
        
        # 도메인 특성
        features["domain_economics"] = 1.0 if domain == "economics" else 0.0
        features["domain_finance"] = 1.0 if domain == "finance" else 0.0
        
        # 검색 결과 특성
        if search_results:
            scores = [result.get("score", 0) for result in search_results[:10]]
            features["max_score"] = max(scores)
            features["mean_score"] = np.mean(scores)
            features["score_std"] = np.std(scores)
        else:
            features["max_score"] = 0.0
            features["mean_score"] = 0.0
            features["score_std"] = 0.0
        
        return features
    
    def _rule_based_threshold(self, features: Dict[str, float], base_threshold: float) -> float:
        """규칙 기반 임계값 계산"""
        threshold = base_threshold
        
        # 도메인별 조정
        if features["domain_economics"] > 0.5:
            threshold -= 0.05
        
        # 쿼리 복잡도에 따른 조정
        if features["query_complexity"] > 0.5:
            threshold -= 0.03
        elif features["query_complexity"] < 0.2:
            threshold += 0.03
        
        # 경제 용어 밀도에 따른 조정
        if features["has_economic_terms"] > 0.3:
            threshold -= 0.04
        
        # 검색 결과 품질에 따른 조정
        if features["score_std"] > 0.15:
            threshold += 0.02
        
        # 임계값 범위 제한
        threshold = max(0.2, min(0.7, threshold))
        
        return threshold
    
    def _count_economic_terms(self, query: str) -> int:
        """경제 용어 개수 계산"""
        query_lower = query.lower()
        return sum(1 for term in ECONOMIC_TERMS if term in query_lower)

class LocalGemmaQAGenerator:
    """Gemma-3-12b-it 기반 로컬 합성 Q&A 생성기"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 도메인별 프롬프트 템플릿
        self.domain_prompts = {
            "economics": {
                "system": "당신은 경제 전문가입니다. 경제 지표와 시장 동향에 대한 질문과 답변을 생성해주세요.",
                "topics": ["소비자물가", "고용률", "GDP", "금리", "인플레이션"]
            },
            "healthcare": {
                "system": "당신은 의료 전문가입니다. 건강과 의료에 대한 질문과 답변을 생성해주세요.",
                "topics": ["질병예방", "건강검진", "영양관리", "운동요법", "스트레스관리"]
            },
            "legal": {
                "system": "당신은 법률 전문가입니다. 법률과 권리에 대한 질문과 답변을 생성해주세요.",
                "topics": ["계약법", "민법", "형법", "노동법", "부동산법"]
            }
        }
    
    def load_model_if_needed(self):
        """필요시 모델 로드"""
        if self.model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                logger.info("Gemma 모델 로딩 중...")
                model_name = "google/gemma-3-12b-it"
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info("✅ Gemma 모델 로딩 완료")
                
            except Exception as e:
                logger.error(f"Gemma 모델 로딩 실패: {e}")
                raise
    
    def generate_qa_pairs(self, domain: str, count: int = 5) -> List[Dict[str, str]]:
        """도메인별 Q&A 쌍 생성"""
        if domain not in self.domain_prompts:
            logger.warning(f"지원하지 않는 도메인: {domain}, 기본 경제 도메인 사용")
            domain = "economics"
        
        self.load_model_if_needed()
        
        qa_pairs = []
        domain_config = self.domain_prompts[domain]
        
        for i in range(count):
            try:
                # 주제 랜덤 선택
                import random
                topic = random.choice(domain_config["topics"])
                
                # 프롬프트 생성
                prompt = f"""<start_of_turn>user
{domain_config['system']}

주제: {topic}

다음 형식으로 1개의 질문과 답변을 생성해주세요:

질문: [구체적인 질문]
답변: [도움이 되는 답변]
<end_of_turn>
<start_of_turn>model
"""
                
                # 텍스트 생성
                generated_text = self._generate_text(prompt)
                
                # Q&A 파싱
                qa_pair = self._parse_qa(generated_text, domain, topic)
                if qa_pair:
                    qa_pairs.append(qa_pair)
                    logger.debug(f"Q&A 생성 {i+1}/{count}: {qa_pair['question'][:30]}...")
                
            except Exception as e:
                logger.error(f"Q&A 생성 실패 {i+1}: {e}")
                continue
        
        logger.info(f"✅ {domain} 도메인 Q&A {len(qa_pairs)}개 생성 완료")
        return qa_pairs
    
    def _generate_text(self, prompt: str) -> str:
        """텍스트 생성"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"텍스트 생성 실패: {e}")
            return ""
    
    def _parse_qa(self, text: str, domain: str, topic: str) -> Optional[Dict[str, str]]:
        """생성된 텍스트에서 Q&A 파싱"""
        try:
            import re
            
            # 질문 추출
            question_match = re.search(r'질문:\s*(.+?)(?=답변:|$)', text, re.DOTALL)
            # 답변 추출
            answer_match = re.search(r'답변:\s*(.+?)(?=$)', text, re.DOTALL)
            
            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()
                
                if len(question) > 5 and len(answer) > 10:
                    return {
                        "question": question,
                        "answer": answer,
                        "domain": domain,
                        "topic": topic,
                        "generated_at": time.time()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Q&A 파싱 실패: {e}")
            return None

class FlyWheelMetricsCollector:
    """플라이휠 워크플로우 메트릭 수집기"""
    
    def __init__(self):
        self.metrics = {
            "keyword_performance": [],
            "weight_performance": [],
            "threshold_performance": [],
            "overall_performance": []
        }
        
        # Gemma 기반 Q&A 생성기 추가
        self.qa_generator = LocalGemmaQAGenerator()
    
    def record_query_performance(self, query: str, result: Dict[str, Any], 
                               user_feedback: float = None):
        """쿼리 성능 기록"""
        metric_record = {
            "timestamp": time.time(),
            "query": query,
            "confidence": result.get("confidence", 0.0),
            "sources_count": len(result.get("sources", [])),
            "answer_length": len(result.get("answer", "")),
            "user_feedback": user_feedback
        }
        
        self.metrics["overall_performance"].append(metric_record)
        
        # 메트릭 크기 제한
        for key in self.metrics:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-800:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        if not self.metrics["overall_performance"]:
            return {"message": "충분한 데이터가 없습니다"}
        
        recent_data = self.metrics["overall_performance"][-100:]
        
        summary = {
            "total_queries": len(self.metrics["overall_performance"]),
            "recent_avg_confidence": np.mean([r["confidence"] for r in recent_data]),
            "recent_avg_sources": np.mean([r["sources_count"] for r in recent_data]),
            "performance_trend": self._calculate_performance_trend()
        }
        
        return summary
    
    def _calculate_performance_trend(self) -> str:
        """성능 트렌드 계산"""
        if len(self.metrics["overall_performance"]) < 20:
            return "insufficient_data"
        
        recent_20 = self.metrics["overall_performance"][-20:]
        prev_20 = self.metrics["overall_performance"][-40:-20]
        
        recent_conf = np.mean([r["confidence"] for r in recent_20])
        prev_conf = np.mean([r["confidence"] for r in prev_20])
        
        if recent_conf > prev_conf + 0.05:
            return "improving"
        elif recent_conf < prev_conf - 0.05:
            return "declining"
        else:
            return "stable"


class ABTestManager:
    """A/B 테스트 관리자 - 기존 시스템 vs 적응형 시스템 성능 비교"""
    
    def __init__(self):
        self.test_data = {
            "variant_A": {"queries": [], "results": []},  # 기존 시스템
            "variant_B": {"queries": [], "results": []}   # 적응형 시스템
        }
        self.user_sessions = {}  # 사용자별 할당된 variant 추적
        
        # A/B 테스트 설정 로드
        try:
            from config import Config
            self.enable_ab_test = Config.ENABLE_AB_TESTING
            self.test_ratio = Config.AB_TEST_RATIO
            self.variants = Config.AB_TEST_VARIANTS
            logger.info(f"✅ A/B 테스트 활성화: {self.enable_ab_test}")
        except Exception as e:
            logger.warning(f"A/B 테스트 설정 로드 실패: {e}")
            self.enable_ab_test = False
    
    def assign_user_variant(self, user_id: str = None) -> str:
        """사용자에게 A/B 테스트 variant 할당"""
        if not self.enable_ab_test:
            return "B"  # A/B 테스트 비활성화시 적응형 사용
        
        if user_id is None:
            user_id = str(uuid.uuid4())[:8]
        
        if user_id in self.user_sessions:
            return self.user_sessions[user_id]
        
        # 50:50 분할
        variant = "A" if random.random() < self.test_ratio else "B"
        self.user_sessions[user_id] = variant
        
        variant_name = self.variants.get(variant, {}).get("name", variant)
        logger.debug(f"사용자 {user_id} → Variant {variant} ({variant_name})")
        return variant
    
    def record_test_result(self, user_id: str, query: str, variant: str, 
                          result: Dict[str, Any], user_feedback: float = None):
        """A/B 테스트 결과 기록"""
        test_record = {
            "timestamp": time.time(),
            "user_id": user_id,
            "query": query,
            "variant": variant,
            "confidence": result.get("confidence", 0.0),
            "response_time": result.get("response_time", 0.0),
            "domain": result.get("domain", "unknown"),
            "answer_length": len(result.get("answer", "")),
            "sources_count": len(result.get("sources", [])),
            "user_feedback": user_feedback
        }
        
        if variant in self.test_data:
            self.test_data[variant]["results"].append(test_record)
            
            # 메모리 관리
            if len(self.test_data[variant]["results"]) > 1000:
                self.test_data[variant]["results"] = self.test_data[variant]["results"][-800:]
    
    def get_ab_test_report(self) -> Dict[str, Any]:
        """A/B 테스트 성과 보고서"""
        if not self.enable_ab_test:
            return {"status": "disabled"}
        
        variant_stats = {}
        
        for variant, data in self.test_data.items():
            if not data["results"]:
                variant_stats[variant] = {"status": "no_data"}
                continue
            
            results = data["results"]
            recent_results = results[-100:] if len(results) > 100 else results
            
            variant_stats[variant] = {
                "name": self.variants.get(variant, {}).get("name", variant),
                "total_queries": len(results),
                "avg_confidence": np.mean([r["confidence"] for r in recent_results]),
                "avg_response_time": np.mean([r["response_time"] for r in recent_results]),
                "user_satisfaction": self._calculate_user_satisfaction(recent_results)
            }
        
        # 승자 결정
        winner = self._determine_winner(variant_stats)
        
        return {
            "status": "active",
            "total_users": len(self.user_sessions),
            "variant_stats": variant_stats,
            "winner": winner,
            "recommendations": self._generate_recommendations(winner)
        }
    
    def _calculate_user_satisfaction(self, results: List[Dict]) -> float:
        """사용자 만족도 계산"""
        feedback_results = [r for r in results if r["user_feedback"] is not None]
        if not feedback_results:
            return 0.0
        return np.mean([r["user_feedback"] for r in feedback_results])
    
    def _determine_winner(self, variant_stats: Dict) -> Dict[str, Any]:
        """승자 결정"""
        if "variant_A" not in variant_stats or "variant_B" not in variant_stats:
            return {"winner": None, "reason": "insufficient_data"}
        
        stats_a = variant_stats["variant_A"]
        stats_b = variant_stats["variant_B"]
        
        if stats_a.get("status") == "no_data" or stats_b.get("status") == "no_data":
            return {"winner": None, "reason": "no_data"}
        
        # 종합 점수 계산
        score_a = (stats_a["avg_confidence"] * 0.5 + 
                  stats_a["user_satisfaction"] * 0.3 +
                  (1/max(stats_a["avg_response_time"], 0.1)) * 0.2)
        
        score_b = (stats_b["avg_confidence"] * 0.5 + 
                  stats_b["user_satisfaction"] * 0.3 +
                  (1/max(stats_b["avg_response_time"], 0.1)) * 0.2)
        
        # 최소 30개 샘플 필요
        if stats_a["total_queries"] < 30 or stats_b["total_queries"] < 30:
            return {"winner": None, "reason": "insufficient_samples"}
        
        improvement_threshold = 0.05
        if score_b > score_a + improvement_threshold:
            return {"winner": "B", "improvement": f"{((score_b-score_a)/score_a*100):.1f}%"}
        elif score_a > score_b + improvement_threshold:
            return {"winner": "A", "improvement": f"{((score_a-score_b)/score_b*100):.1f}%"}
        else:
            return {"winner": "tie", "reason": "no_significant_difference"}
    
    def _generate_recommendations(self, winner: Dict) -> List[str]:
        """개선 권장사항"""
        recommendations = []
        
        if winner.get("winner") == "B":
            recommendations.append("✅ 적응형 시스템으로 전환 권장")
            recommendations.append("💡 플라이휠 워크플로우 효과 확인됨")
        elif winner.get("winner") == "A":
            recommendations.append("⚠️ 적응형 시스템 파라미터 튜닝 필요")
        else:
            recommendations.append("📊 더 많은 데이터 수집 필요")
        
        return recommendations 