#!/usr/bin/env python3
"""
도메인별 처리 방식 상세 데모
의료, 법률, 경제 도메인에서 시스템이 어떻게 적응하는지 보여줍니다
"""
from adaptive_rag_components import (
    DomainDetector, 
    AdaptiveKeywordExtractor, 
    AdaptiveWeightCalculator,
    SmartThresholdCalculator
)
import numpy as np

def demonstrate_domain_processing():
    """도메인별 처리 과정 시연"""
    
    # 테스트 쿼리들 (도메인별)
    test_scenarios = {
        "economics": {
            "queries": [
                "미국의 소비자물가지수(CPI) 상승률이 경제에 미치는 영향을 분석해주세요",
                "한국은행의 기준금리 인상 결정 배경은 무엇인가요?",
                "최근 고용시장 동향과 실업률 변화를 설명해주세요"
            ],
            "expected_keywords": ["소비자물가", "CPI", "경제", "한국은행", "기준금리", "고용", "실업률"],
            "expected_weights": {"vector": 0.65, "rerank": 0.35},
            "expected_threshold": 0.42  # 경제는 더 관대
        },
        
        "healthcare": {
            "queries": [
                "당뇨병 환자의 혈당 관리 방법에 대해 알려주세요",
                "고혈압 예방을 위한 생활습관 개선 방안은?",
                "코로나19 백신 접종 후 주의사항을 설명해주세요"
            ],
            "expected_keywords": ["당뇨병", "혈당", "관리", "고혈압", "예방", "생활습관", "코로나19", "백신"],
            "expected_weights": {"vector": 0.6, "rerank": 0.4},  # 의료는 균형
            "expected_threshold": 0.5   # 의료는 더 엄격
        },
        
        "legal": {
            "queries": [
                "임대차 계약서 작성 시 주의해야 할 법적 사항은?",
                "교통사고 발생 시 보험금 청구 절차를 알려주세요",
                "직장 내 괴롭힘 신고 방법과 구제 절차는?"
            ],
            "expected_keywords": ["임대차", "계약서", "법적", "교통사고", "보험금", "괴롭힘", "신고", "구제"],
            "expected_weights": {"vector": 0.55, "rerank": 0.45},  # 법률은 리랭킹 중요
            "expected_threshold": 0.48
        }
    }
    
    # 컴포넌트 초기화
    domain_detector = DomainDetector()
    
    # Mock 임베딩 모델 (실제 사용시에는 진짜 모델 필요)
    class MockEmbeddingModel:
        def encode(self, texts):
            return np.random.rand(len(texts), 384)
    
    keyword_extractor = AdaptiveKeywordExtractor(MockEmbeddingModel())
    weight_calculator = AdaptiveWeightCalculator()
    threshold_calculator = SmartThresholdCalculator()
    
    print("🎯 도메인별 적응형 처리 시연")
    print("=" * 80)
    
    for domain_name, scenario in test_scenarios.items():
        print(f"\n📋 {domain_name.upper()} 도메인 처리")
        print("-" * 60)
        
        for i, query in enumerate(scenario["queries"], 1):
            print(f"\n{i}. 쿼리: {query}")
            
            # 1. 도메인 감지
            detected_domain = domain_detector.detect_domain(query)
            print(f"   감지된 도메인: {detected_domain}")
            
            # 2. 적응형 키워드 추출
            keyword_result = keyword_extractor.extract_keywords_adaptive(query, domain=detected_domain)
            extracted_keywords = keyword_result.get('final_keywords', [])
            print(f"   추출된 키워드: {extracted_keywords}")
            
            # 3. 가상의 검색 결과 생성 (도메인별 특성 반영)
            mock_search_results = generate_mock_search_results(detected_domain, query)
            
            # 4. 적응형 가중치 계산
            vector_weight, rerank_weight = weight_calculator.calculate_adaptive_weights(
                query, mock_search_results, detected_domain
            )
            print(f"   적응형 가중치: 벡터 {vector_weight:.3f}, 리랭킹 {rerank_weight:.3f}")
            
            # 5. 스마트 임계값 계산
            smart_threshold = threshold_calculator.calculate_smart_threshold(
                query, mock_search_results, detected_domain
            )
            print(f"   스마트 임계값: {smart_threshold:.3f}")
            
            # 6. 도메인별 특성 분석
            domain_analysis = analyze_domain_specific_features(query, detected_domain, keyword_result)
            print(f"   도메인 특성: {domain_analysis}")
            
            print()
    
    # 도메인별 특성 요약
    print("\n📊 도메인별 처리 특성 요약")
    print("=" * 80)
    
    domain_characteristics = {
        "economics": {
            "특징": "수치 데이터와 트렌드 분석 중심",
            "키워드_유형": "지표명, 정책용어, 수치표현",
            "가중치_전략": "벡터 유사도 우선 (정확한 지표 매칭 중요)",
            "임계값_전략": "상대적으로 관대 (경제 용어 풍부)",
            "신뢰도_요소": "정확한 수치, 시점, 출처 중요"
        },
        "healthcare": {
            "특징": "전문 의학용어와 증상 설명 중심",
            "키워드_유형": "질병명, 증상, 치료법, 예방법",
            "가중치_전략": "균형적 접근 (전문성과 문맥 모두 중요)",
            "임계값_전략": "엄격 적용 (정확성 최우선)",
            "신뢰도_요소": "의학적 정확성, 안전성 정보 필수"
        },
        "legal": {
            "특징": "법률 조항과 절차 설명 중심",
            "키워드_유형": "법률용어, 절차명, 권리의무",
            "가중치_전략": "리랭킹 중시 (문맥적 해석 중요)",
            "임계값_전략": "중간 수준 (전문성과 접근성 균형)",
            "신뢰도_요소": "법적 근거, 절차 정확성, 최신성"
        }
    }
    
    for domain, chars in domain_characteristics.items():
        print(f"\n🎯 {domain.upper()}:")
        for key, value in chars.items():
            print(f"   {key}: {value}")

def generate_mock_search_results(domain: str, query: str):
    """도메인별 가상 검색 결과 생성"""
    
    domain_content = {
        "economics": [
            {"score": 0.85, "content": "한국은행이 발표한 소비자물가지수(CPI)는 전년 동월 대비 3.2% 상승했다. 이는 에너지 가격 상승과 식료품 가격 인상이 주요 원인으로 분석된다."},
            {"score": 0.78, "content": "미국 연방준비제도(Fed)의 기준금리 인상 결정은 글로벌 경제에 파급효과를 미치고 있다. 신흥국 자본 유출과 환율 변동성이 확대되고 있다."},
            {"score": 0.72, "content": "고용률은 전월 대비 0.2%p 상승한 62.1%를 기록했다. 특히 서비스업 중심으로 일자리가 증가하면서 실업률도 2.8%로 하락했다."}
        ],
        "healthcare": [
            {"score": 0.82, "content": "당뇨병 환자의 혈당 관리를 위해서는 규칙적인 식사와 운동이 필수적이다. 혈당 측정기를 활용한 자가 모니터링도 중요하다."},
            {"score": 0.76, "content": "고혈압 예방을 위해서는 저염식 식단, 규칙적인 유산소 운동, 금연, 금주가 권장된다. 정기적인 혈압 측정도 필요하다."},
            {"score": 0.70, "content": "코로나19 백신 접종 후에는 15-30분간 접종 장소에서 대기하며 이상반응을 관찰해야 한다. 발열, 근육통 등은 일반적인 반응이다."}
        ],
        "legal": [
            {"score": 0.80, "content": "임대차 계약서에는 임대료, 보증금, 계약기간, 특약사항을 명확히 기재해야 한다. 전월세 상한제와 계약갱신청구권도 확인해야 한다."},
            {"score": 0.74, "content": "교통사고 발생 시에는 경찰신고, 보험사 신고, 병원 진료를 순서대로 진행해야 한다. 사고 현장 사진과 상대방 정보 수집이 중요하다."},
            {"score": 0.68, "content": "직장 내 괴롭힘 신고는 회사 내부 신고센터나 고용노동부를 통해 가능하다. 증거 수집과 피해 기록 작성이 선행되어야 한다."}
        ]
    }
    
    return domain_content.get(domain, domain_content["economics"])

def analyze_domain_specific_features(query: str, domain: str, keyword_result: Dict):
    """도메인별 특성 분석"""
    
    features = {
        "economics": {
            "수치_포함": any(char.isdigit() for char in query),
            "정책_용어": any(term in query for term in ["정책", "금리", "인상", "인하"]),
            "지표_언급": any(term in query for term in ["CPI", "GDP", "고용률", "실업률"]),
            "키워드_수": len(keyword_result.get('final_keywords', []))
        },
        "healthcare": {
            "질병_언급": any(term in query for term in ["병", "질환", "증상", "치료"]),
            "예방_관련": any(term in query for term in ["예방", "관리", "주의"]),
            "의료_행위": any(term in query for term in ["진료", "검사", "처방", "수술"]),
            "키워드_수": len(keyword_result.get('final_keywords', []))
        },
        "legal": {
            "법률_용어": any(term in query for term in ["법", "계약", "권리", "의무"]),
            "절차_관련": any(term in query for term in ["절차", "방법", "신고", "청구"]),
            "분쟁_관련": any(term in query for term in ["사고", "분쟁", "피해", "구제"]),
            "키워드_수": len(keyword_result.get('final_keywords', []))
        }
    }
    
    return features.get(domain, {"일반_쿼리": True})

def compare_domain_adaptations():
    """도메인별 적응 방식 비교"""
    
    print("\n⚖️ 도메인별 적응 방식 상세 비교")
    print("=" * 80)
    
    # 동일한 질문을 다른 도메인으로 처리했을 때의 차이점
    universal_query = "최근 동향과 전망에 대해 설명해주세요"
    
    domain_contexts = {
        "economics": "경제 지표와 시장 동향",
        "healthcare": "질병 발생률과 건강 트렌드", 
        "legal": "법률 개정과 판례 동향"
    }
    
    print(f"공통 질문: '{universal_query}'")
    print("\n각 도메인별 해석과 처리 방식:")
    
    for domain, context in domain_contexts.items():
        print(f"\n🎯 {domain.upper()} 도메인 처리:")
        print(f"   - 컨텍스트 해석: {context}")
        print(f"   - 예상 키워드: {get_domain_specific_keywords(domain)}")
        print(f"   - 가중치 전략: {get_domain_weight_strategy(domain)}")
        print(f"   - 임계값 적용: {get_domain_threshold_strategy(domain)}")
        print(f"   - 신뢰도 기준: {get_domain_confidence_criteria(domain)}")

def get_domain_specific_keywords(domain: str) -> List[str]:
    """도메인별 예상 키워드"""
    keywords = {
        "economics": ["동향", "전망", "경제", "지표", "성장", "시장"],
        "healthcare": ["동향", "전망", "건강", "질병", "예방", "치료"],
        "legal": ["동향", "전망", "법률", "개정", "판례", "제도"]
    }
    return keywords.get(domain, ["동향", "전망"])

def get_domain_weight_strategy(domain: str) -> str:
    """도메인별 가중치 전략"""
    strategies = {
        "economics": "벡터 우선 (65%) - 정확한 지표 매칭 중요",
        "healthcare": "균형 접근 (60% vs 40%) - 전문성과 문맥 조화",
        "legal": "리랭킹 강화 (55% vs 45%) - 법적 해석 중요"
    }
    return strategies.get(domain, "기본 전략")

def get_domain_threshold_strategy(domain: str) -> str:
    """도메인별 임계값 전략"""
    strategies = {
        "economics": "관대한 임계값 (0.42) - 경제 용어 풍부",
        "healthcare": "엄격한 임계값 (0.50) - 정확성 최우선",
        "legal": "중간 임계값 (0.48) - 전문성과 접근성 균형"
    }
    return strategies.get(domain, "기본 임계값")

def get_domain_confidence_criteria(domain: str) -> str:
    """도메인별 신뢰도 기준"""
    criteria = {
        "economics": "수치 정확성, 시점 명시, 출처 신뢰성",
        "healthcare": "의학적 정확성, 안전성 고려, 전문가 검증",
        "legal": "법적 근거, 최신 법령, 절차 정확성"
    }
    return criteria.get(domain, "일반 기준")

if __name__ == "__main__":
    print("🚀 도메인별 적응형 처리 시스템 데모")
    
    # 1. 기본 도메인 처리 시연
    demonstrate_domain_processing()
    
    # 2. 도메인 적응 방식 비교
    compare_domain_adaptations()
    
    print("\n✅ 도메인별 처리 데모 완료!")
    print("\n💡 핵심 포인트:")
    print("   - 각 도메인별로 키워드, 가중치, 임계값이 자동 조정됩니다")
    print("   - 의료는 정확성, 법률은 해석, 경제는 데이터 매칭을 우선시합니다")
    print("   - 새로운 도메인 추가도 간단히 확장 가능합니다") 