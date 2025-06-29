# multimodalrag

🔧 핵심 기술 스택
임베딩 모델: dragonkue/BGE-m3-ko (한국어 특화, 768차원)
리랭커: dragonkue/bge-reranker-v2-m3-ko
LLM: google/gemma-3-12b-it (답변 생성용)
벡터 DB: Qdrant
NLP: KiwiPiepy (한국어 형태소 분석)
API: FastAPI (백엔드) + Streamlit (프론트엔드)


 🔄 RAG 워크플로우 다이어그램

 graph TD
    A["🔍 사용자 질문<br/>Query Input"] --> B["📏 쿼리 임베딩<br/>BGE-m3-ko Embedding<br/>768차원 벡터"]
    
    B --> C["🔎 벡터 검색<br/>Qdrant Vector Search<br/>TOP_K=10<br/>Cosine Similarity"]
    
    C --> D{"🎯 문서 연관성 판단<br/>Dynamic Threshold<br/>+ Keyword Boost<br/>+ Semantic Boost"}
    
    D -->|"연관성 없음"| E["⚠️ 유도 답변 생성<br/>Guide to Relevant Topics"]
    D -->|"연관성 있음"| F["📊 질문 의도 분석<br/>Query Intent Analysis<br/>+ 도메인 감지"]
    
    F --> G["🔄 리랭킹<br/>BGE-reranker-v2-m3-ko<br/>TOP_K=5<br/>정밀 재순위"]
    
    G --> H["🎯 관련 질문 검색<br/>Related Questions<br/>from Questions Section"]
    
    H --> I["📝 컨텍스트 구성<br/>Context Preparation<br/>+ Source Information<br/>+ 메타데이터"]
    
    I --> J["🤖 답변 생성<br/>Gemma-3-12b-it<br/>Multi-GPU 추론<br/>+ 출처 정보 포함"]
    
    J --> K["📊 신뢰도 계산<br/>Vector Score 60%<br/>+ Rerank Score 40%<br/>+ 도메인 가중치"]
    
    K --> L["✅ 최종 답변<br/>Answer + Sources<br/>+ Confidence + Related Q"]
    
    E --> L
    
    subgraph "🗂️ 데이터 준비 프로세스"
        M["📄 마크다운 문서<br/>Text + Image Analysis"] --> N["🔧 문서 파싱<br/>DocumentLoader<br/>청크 1000자/오버랩 200자"]
        N --> O["🧮 임베딩 생성<br/>BGE-m3-ko<br/>768차원 벡터"]
        O --> P["💾 벡터 저장<br/>Qdrant Database<br/>Collection: multimodal_rag"]
    end
    
    subgraph "🧠 적응형 키워드 추출"
        Q["📝 형태소 분석<br/>KiwiPiepy<br/>명사/동사/형용사"] --> R["🎯 의미적 확장<br/>Semantic Expansion<br/>도메인별 Seed Keywords"]
        R --> S["📈 TF-IDF 분석<br/>Importance Scoring<br/>문서 집합 기반"]
        S --> T["🔗 키워드 통합<br/>Final Keywords<br/>품질 필터링"]
    end
    
    subgraph "🔄 플라이휠 워크플로우"
        U["📊 데이터 준비<br/>Ground Truth 분석"] --> V["🧠 모델 최적화<br/>적응형 컴포넌트 튜닝"]
        V --> W["📈 성능 평가<br/>신뢰도/처리시간 측정"]
        W --> X["🔄 데이터 강화<br/>합성 Q&A 생성"]
        X --> U
    end
    
    style A fill:#ffeb3b
    style D fill:#ff5722
    style G fill:#9c27b0
    style J fill:#4caf50
    style L fill:#2196f3

RETRIEVE (검색) 단계

graph TB
    subgraph "🔍 RETRIEVE 단계 상세"
        A1["사용자 질문 입력<br/>'미국 CPI 동향은?'"] --> B1["📏 임베딩 변환<br/>BGE-m3-ko<br/>768차원 벡터"]
        
        B1 --> C1["🔎 벡터 검색<br/>Qdrant DB<br/>Cosine Similarity<br/>TOP_K=10"]
        
        C1 --> D1["📊 초기 결과<br/>Score > 0.0<br/>10개 문서 후보"]
        
        D1 --> E1{"🎯 연관성 판단<br/>Smart Threshold<br/>기본값: 0.45"}
        
        E1 -->|Pass| F1["✅ 관련 문서<br/>Relevant Docs"]
        E1 -->|Fail| G1["❌ 비관련<br/>→ 유도 답변"]
        
        subgraph "🔧 동적 임계값 계산"
            H1["📏 질문 길이 분석<br/>Length Analysis<br/>길이별 가중치"]
            I1["🔑 키워드 밀도<br/>Economic Keywords<br/>도메인 용어 비율"]
            J1["📈 점수 분산도<br/>Score Variance<br/>검색 결과 일관성"]
            K1["🎯 도메인 적합성<br/>Domain Relevance<br/>경제/금융/일반"]
            
            H1 --> L1["⚖️ 최종 임계값<br/>범위: 0.2-0.7<br/>경제도메인 -0.05 보정"]
            I1 --> L1
            J1 --> L1
            K1 --> L1
        end
        
        subgraph "🚀 성능 최적화"
            M1["💾 키워드 캐싱<br/>1000개 TTL 1시간"]
            N1["🎯 도메인 캐싱<br/>LLM 결과 2시간"]
            O1["⚡ 병렬 처리<br/>Multi-GPU 활용"]
        end
    end
    
    style A1 fill:#ffeb3b
    style E1 fill:#ff5722
    style L1 fill:#9c27b0

 AUGMENT (증강) 단계

 graph TB
    subgraph "🔧 AUGMENT 단계 상세"
        A2["📊 검색 결과<br/>10개 문서 후보"] --> B2["🔄 리랭킹<br/>BGE-reranker-v2-m3-ko<br/>Cross-Encoding"]
        
        B2 --> C2["📋 TOP_K=5<br/>최고 품질 문서"]
        
        C2 --> D2["🧠 적응형 키워드 추출<br/>AdaptiveKeywordExtractor"]
        
        D2 --> E2["📝 컨텍스트 구성<br/>Context Assembly<br/>문서 + 메타데이터"]
        
        E2 --> F2["🎯 관련 질문 검색<br/>Related Questions<br/>벡터 유사도 기반"]
        
        F2 --> G2["✅ 증강된 컨텍스트<br/>Enhanced Context"]
        
        subgraph "🧠 키워드 추출 상세"
            H2["📝 형태소 분석<br/>KiwiPiepy<br/>명사/동사/형용사 추출"]
            I2["🎯 의미적 확장<br/>도메인별 Seed Keywords<br/>경제: CPI, GDP, 인플레이션"]
            J2["📈 TF-IDF 분석<br/>문서 집합 기반<br/>중요도 스코어링"]
            K2["🔗 키워드 통합<br/>품질 필터링<br/>중복 제거"]
            
            H2 --> K2
            I2 --> K2
            J2 --> K2
        end
        
        subgraph "🔄 리랭킹 프로세스"
            L2["🔍 쿼리-문서 페어링<br/>Query-Doc Pairs<br/>5개 문서별 스코어링"]
            M2["🧮 교차 인코딩<br/>Cross-Encoding<br/>정밀한 관련성 측정"]
            N2["📊 관련성 점수<br/>0-1 정규화<br/>Relevance Scores"]
            O2["🏆 순위 재조정<br/>Re-ranking<br/>최종 순서 결정"]
            
            L2 --> M2
            M2 --> N2
            N2 --> O2
        end
        
        subgraph "⚖️ 적응형 가중치"
            P2["🎯 도메인 감지<br/>LLM + 키워드 하이브리드"]
            Q2["📊 쿼리 특성 분석<br/>길이, 복잡도, 의도"]
            R2["🔧 동적 가중치 계산<br/>경제: 65%-35%<br/>금융: 60%-40%<br/>일반: 60%-40%"]
            
            P2 --> R2
            Q2 --> R2
        end
    end
    
    style A2 fill:#ffeb3b
    style D2 fill:#9c27b0
    style G2 fill:#4caf50

GENERATE (생성) 단계

graph TB
    subgraph "🤖 GENERATE 단계 상세"
        A3["📝 증강된 컨텍스트<br/>Enhanced Context<br/>+ 관련 질문"] --> B3["📋 프롬프트 엔지니어링<br/>Prompt Engineering"]
        
        B3 --> C3["🤖 LLM 추론<br/>Gemma-3-12b-it<br/>Multi-GPU 병렬 처리"]
        
        C3 --> D3["📊 신뢰도 계산<br/>Confidence Scoring<br/>벡터 + 리랭킹 가중합"]
        
        D3 --> E3["📚 출처 정보 첨부<br/>Source Attribution<br/>문서 제목 + URL"]
        
        E3 --> F3["✅ 최종 답변<br/>Complete Answer<br/>+ 신뢰도 + 출처"]
        
        subgraph "📋 프롬프트 구성 요소"
            G3["🎯 질문 분석<br/>Query Analysis<br/>의도 파악 + 키워드"]
            H3["📄 컨텍스트 정리<br/>Context Summary<br/>핵심 정보 요약"]
            I3["📚 출처 정보<br/>Source Information<br/>문서 메타데이터"]
            J3["🔧 답변 형식 지정<br/>Response Format<br/>구조화된 답변 템플릿"]
            
            G3 --> K3["📝 완성된 프롬프트<br/>Final Prompt<br/>최대 2048 토큰"]
            H3 --> K3
            I3 --> K3
            J3 --> K3
        end
        
        subgraph "📊 신뢰도 계산 공식"
            L3["🔍 벡터 점수<br/>Vector Score<br/>60-65% 가중치"]
            M3["🔄 리랭킹 점수<br/>Reranking Score<br/>35-40% 가중치"]
            N3["🎯 도메인 보정<br/>Domain Adjustment<br/>±0.05 조정"]
            
            L3 --> O3["🏆 최종 신뢰도<br/>Final Confidence<br/>0-1 범위 정규화"]
            M3 --> O3
            N3 --> O3
        end
        
        subgraph "🚀 LLM 최적화"
            P3["💾 Float16 정밀도<br/>메모리 효율성"]
            Q3["🎮 Multi-GPU 분산<br/>모델 병렬 처리"]
            R3["⚡ 텐서 건전성 체크<br/>NaN/Inf 검증"]
            S3["🧹 GPU 메모리 정리<br/>자동 캐시 클리어"]
        end
        
        subgraph "📝 답변 후처리"
            T3["✂️ 중복 제거<br/>Repetition Removal"]
            U3["📏 길이 조정<br/>적정 길이 유지"]
            V3["🔗 출처 링크 생성<br/>Source URL 포맷"]
            W3["📋 관련 질문 추천<br/>Related Q Suggestions"]
        end
    end
    
    style A3 fill:#ffeb3b
    style C3 fill:#ff5722
    style F3 fill:#2196f3

합성데이터 생성 아키텍처


graph TD
    subgraph "🔄 플라이휠 워크플로우 - 합성데이터 사이클"
        A4["📊 1단계: 데이터 준비<br/>Ground Truth 품질 분석<br/>새 데이터 소스 식별"] --> B4["🧠 2단계: 모델 최적화<br/>적응형 컴포넌트 튜닝<br/>가중치/임계값 최적화"]
        
        B4 --> C4["📈 3단계: 성능 평가<br/>테스트 쿼리 실행<br/>신뢰도/처리시간 측정"]
        
        C4 --> D4["🔄 4단계: 데이터 강화<br/>합성 Q&A 생성<br/>난이도별 질문 생성"]
        
        D4 --> A4
        
        D4 --> E4["📝 합성 Q&A 생성기<br/>LocalSyntheticQAGenerator<br/>Gemma-3-12b-it 기반"]
        
        E4 --> F4{"🎯 도메인별 생성"}
        
        F4 --> G4["💰 경제 도메인<br/>CPI, GDP, 인플레이션<br/>통화정책, 고용률"]
        F4 --> H4["📈 금융 도메인<br/>주식, 채권, 투자전략<br/>포트폴리오, 리스크"]
        F4 --> I4["🏥 의료 도메인<br/>질병예방, 건강관리<br/>치료법, 영양관리"]
        F4 --> J4["⚖️ 법률 도메인<br/>민법, 형법, 계약법<br/>소송절차, 권리구제"]
        
        subgraph "📝 Q&A 생성 프로세스"
            K4["🎲 랜덤 주제 선택<br/>도메인별 Topics<br/>질문 유형 선택"] --> L4["📋 프롬프트 생성<br/>시스템 메시지<br/>+ 주제 + 유형"]
            
            L4 --> M4["🤖 Gemma-3 추론<br/>Temperature: 0.7<br/>Top-p: 0.9"]
            
            M4 --> N4["✂️ Q&A 파싱<br/>정규식 추출<br/>품질 검증"]
            
            N4 --> O4["📊 품질 평가<br/>길이, 완성도<br/>반복성 체크"]
            
            O4 -->|품질 ≥ 0.7| P4["✅ 고품질 Q&A<br/>저장 및 활용"]
            O4 -->|품질 < 0.7| Q4["❌ 저품질<br/>폐기"]
        end
        
        subgraph "🎯 도메인별 특화 설정"
            R4["경제: 최근 동향, 원인 분석<br/>전망, 비교, 정책 영향"]
            S4["금융: 투자 조언, 위험 분석<br/>상품 비교, 시장 전망"]
            T4["의료: 증상 관련, 예방법<br/>치료법, 생활습관"]
            U4["법률: 법적 권리, 절차 안내<br/>계약, 분쟁해결"]
        end
    end
    
    style E4 fill:#ff5722
    style P4 fill:#4caf50
    style Q4 fill:#f44336

 1. 합성데이터 생성 목적

✅ 플라이휠 워크플로우의 핵심 동력
✅ 도메인별 특화 Q&A 데이터 확보  
✅ 모델 성능 지속적 개선
✅ 데이터 부족 문제 해결


생성 파이프라인

단계별 프로세스:
도메인 선택 → 경제/금융/의료/법률 중 선택
주제 랜덤 선택 → 각 도메인별 10개 주제 풀에서 선택
질문 유형 선택 → 5가지 질문 유형 중 랜덤 선택
프롬프트 구성 → 시스템 메시지 + 주제 + 유형
Gemma-3 추론 → 고품질 Q&A 생성
품질 검증 → 길이, 완성도, 반복성 체크
저장/폐기 → 품질 0.7 이상만 저장



4단계 플라이휠 워크플로우:

데이터 준비 → Ground Truth 품질 분석
모델 최적화 → 적응형 컴포넌트 튜닝
성능 평가 → 실제 성능 측정
데이터 강화 → 합성 Q&A 생성으로 데이터셋 확장