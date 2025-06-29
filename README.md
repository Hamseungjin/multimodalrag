# multimodalrag

## 🔧 핵심 기술 스택
1. 임베딩 모델: dragonkue/BGE-m3-ko (한국어 특화, 768차원)
2. 리랭커: dragonkue/bge-reranker-v2-m3-ko
3. LLM: google/gemma-3-12b-it (답변 생성용)
4. 벡터 DB: Qdrant
5. NLP: KiwiPiepy (한국어 형태소 분석)
6. API: FastAPI (백엔드) + Streamlit (프론트엔드)

# Multi Modal RAG 워크플로우

```mermaid
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

    %% 데이터 준비 프로세스
    subgraph "🗂️ 데이터 준비 프로세스"
        M["📄 마크다운 문서<br/>Text + Image Analysis"] --> N["🔧 문서 파싱<br/>DocumentLoader<br/>청크 1000자/오버랩 200자"]
        N --> O["🧮 임베딩 생성<br/>BGE-m3-ko<br/>768차원 벡터"]
        O --> P["💾 벡터 저장<br/>Qdrant Database<br/>Collection: multimodal_rag"]
    end

    %% 적응형 키워드 추출
    subgraph "🧠 적응형 키워드 추출"
        Q["📝 형태소 분석<br/>KiwiPiepy<br/>명사/동사/형용사"] --> R["🎯 의미적 확장<br/>Semantic Expansion<br/>도메인별 Seed Keywords"]
        R --> S["📈 TF-IDF 분석<br/>Importance Scoring<br/>문서 집합 기반"]
        S --> T["🔗 키워드 통합<br/>Final Keywords<br/>품질 필터링"]
    end

    %% 플라이휠 워크플로우
    subgraph "🔄 플라이휠 워크플로우"
        U["📊 데이터 준비<br/>Ground Truth 분석"] --> V["🧠 모델 최적화<br/>적응형 컴포넌트 튜닝"]
        V --> W["📈 성능 평가<br/>신뢰도/처리시간 측정"]
        W --> X["🔄 데이터 강화<br/>합성 Q&A 생성"]
        X --> U
    end

    %% 스타일 지정
    style A fill:#ffeb3b
    style D fill:#ff5722
    style G fill:#9c27b0
    style J fill:#4caf50
    style L fill:#2196f3
```

# 합성데이터 워크플로우

```mermaid
graph TD
A["📊 데이터 준비<br/>Ground Truth 품질 분석"] --> B["🧠 모델 최적화<br/>가중치/임계값 튜닝"]
B --> C["📈 성능 평가<br/>신뢰도/속도 측정"]
C --> D["🔄 데이터 강화<br/>합성 Q&A 생성"]
D --> E["📝 합성 Q&A 생성기<br/>Gemma-3 기반"]
E --> F{"🎯 도메인별 분기"}
F --> G1["💰 경제<br/>CPI, GDP 등"]
F --> G2["📈 금융<br/>주식, 채권 등"]
F --> G3["🏥 의료<br/>질병, 예방 등"]
F --> G4["⚖️ 법률<br/>계약, 절차 등"]
G1 --> H["🧪 품질 검증<br/>≥0.7 저장<br/><0.7 폐기"]
G2 --> H
G3 --> H
G4 --> H
H --> A
```

## 합성데이터 생성 목적

1. 플라이휠 워크플로우의 핵심 동력
2. 도메인별 특화 Q&A 데이터 확보  
3. 모델 성능 지속적 개선
4. 데이터 부족 문제 해결

