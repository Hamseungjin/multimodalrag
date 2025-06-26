
## 1. 코드베이스 분석 📊

핵심 워크플로우
graph LR
    subgraph "🎯 입력 단계"
        A["👤 사용자 질문<br/>'미국 CPI 동향은?'"]
    end
    
    subgraph "🔍 RETRIEVE (검색)"
        B["📏 임베딩 변환<br/>BGE-m3-ko<br/>768차원 벡터"]
        C["🔎 벡터 검색<br/>Qdrant DB<br/>TOP 10 후보"]
        D["🎯 동적 임계값 판단<br/>• 질문 길이 분석<br/>• 키워드 밀도 계산<br/>• 연관성 스코어링"]
    end
    
    subgraph "🔧 AUGMENT (증강)"
        E["🔄 리랭킹<br/>BGE-reranker<br/>TOP 5 정밀 선별"]
        F["🔑 키워드 추출<br/>• 형태소 분석<br/>• 의미적 확장<br/>• TF-IDF 가중치"]
        G["📝 컨텍스트 구성<br/>문서 + 메타데이터<br/>+ 출처 정보"]
    end
    
    subgraph "🤖 GENERATE (생성)"
        H["🎨 프롬프트 엔지니어링<br/>• 질문 분석<br/>• 컨텍스트 요약<br/>• 출처 형식 지정"]
        I["🧠 답변 생성<br/>Gemma-3-12b-it<br/>Multi-GPU 추론"]
        J["📊 신뢰도 계산<br/>벡터(60%) + 리랭킹(40%)<br/>→ 0-1 점수"]
    end
    
    subgraph "✅ 출력 단계"
        K["📋 최종 답변<br/>• 상세한 답변<br/>• 출처 문서 목록<br/>• 신뢰도 점수<br/>• 관련 질문 추천"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    
    style A fill:#ffeb3b
    style D fill:#ff5722
    style F fill:#9c27b0
    style I fill:#4caf50
    style K fill:#2196f3


### 🏗️ **전체 시스템 아키텍처**

**MultiModalRAG**는 한국은행 뉴스 데이터를 활용한 고급 RAG 시스템으로, 다음과 같은 구성요소로 이루어져 있습니다:

### 📁 **주요 파일 구조**
- **`config.py`**: 시스템 설정 및 하이퍼파라미터
- **`rag_utils.py`**: 핵심 RAG 유틸리티 클래스들 (2302줄)
- **`embed_data.py`**: 데이터 임베딩 및 벡터 저장 스크립트
- **`main.py`**: FastAPI 백엔드 서버
- **`app.py`**: Streamlit 프론트엔드 인터페이스

### 🔧 **핵심 기술 스택**
- **임베딩 모델**: `dragonkue/BGE-m3-ko` (한국어 특화)
- **리랭커**: `dragonkue/bge-reranker-v2-m3-ko`
- **LLM**: `google/gemma-3-12b-it`
- **벡터 DB**: Qdrant
- **NLP**: KiwiPiepy (한국어 형태소 분석)

### 🎯 **특별한 기능들**
1. **동적 임계값 조정**: 질문 길이와 키워드 밀도에 따른 적응형 임계값
2. **고급 키워드 추출**: 형태소 분석 + 의미적 확장 + TF-IDF
3. **멀티모달 지원**: 텍스트 + 이미지 분석 결과 통합
4. **문서 연관성 판단**: 비관련 질문 시 유도 답변 제공

## 2. RAG 워크플로우 다이어그램 🔄
graph TD
    A["🔍 사용자 질문<br/>Query Input"] --> B["📏 쿼리 임베딩<br/>BGE-m3-ko Embedding"]
    
    B --> C["🔎 벡터 검색<br/>Qdrant Vector Search<br/>(TOP_K=10)"]
    
    C --> D{"🎯 문서 연관성 판단<br/>Dynamic Threshold<br/>+ Keyword Boost<br/>+ Semantic Boost"}
    
    D -->|"연관성 없음"| E["⚠️ 유도 답변 생성<br/>Guide to Relevant Topics"]
    D -->|"연관성 있음"| F["📊 질문 의도 분석<br/>Query Intent Analysis"]
    
    F --> G["🔄 리랭킹<br/>BGE-reranker-v2-m3-ko<br/>(TOP_K=5)"]
    
    G --> H["🎯 관련 질문 검색<br/>Related Questions<br/>from Questions Section"]
    
    H --> I["📝 컨텍스트 구성<br/>Context Preparation<br/>+ Source Information"]
    
    I --> J["🤖 답변 생성<br/>Gemma-3-12b-it<br/>+ 출처 정보 포함"]
    
    J --> K["📊 신뢰도 계산<br/>Vector Score (60%)<br/>+ Rerank Score (40%)"]
    
    K --> L["✅ 최종 답변<br/>Answer + Sources<br/>+ Confidence + Related Q"]
    
    E --> L
    
    subgraph "🗂️ 데이터 준비 프로세스"
        M["📄 마크다운 문서<br/>Text + Image Analysis"] --> N["🔧 문서 파싱<br/>DocumentLoader"]
        N --> O["🧮 임베딩 생성<br/>BGE-m3-ko"]
        O --> P["💾 벡터 저장<br/>Qdrant Database"]
    end
    
    subgraph "🧠 고급 키워드 추출"
        Q["📝 형태소 분석<br/>KiwiPiepy"] --> R["🎯 의미적 확장<br/>Semantic Expansion"]
        R --> S["📈 TF-IDF 분석<br/>Importance Scoring"]
        S --> T["🔗 키워드 통합<br/>Final Keywords"]
    end
## 3. Retrieve-Augment-Generate 각 단계 상세 분석 🔍

### 🔍 **Retrieve (검색) 단계**
graph LR
    subgraph "1️⃣ RETRIEVE 단계"
        A["🔍 질문 입력"] --> B["📏 임베딩 변환<br/>BGE-m3-ko<br/>768차원 벡터"]
        
        B --> C["🔎 벡터 검색<br/>Qdrant DB<br/>Cosine Similarity"]
        
        C --> D["📊 초기 결과<br/>TOP_K=10<br/>Score > 0.0"]
        
        D --> E{"🎯 연관성 판단<br/>Dynamic Threshold<br/>Base: 0.45"}
        
        E -->|"Pass"| F["✅ 관련 문서<br/>Relevant Docs"]
        E -->|"Fail"| G["❌ 비관련<br/>Non-Relevant"]
        
        subgraph "🔧 동적 임계값 계산"
            H["📏 질문 길이<br/>Length Analysis"]
            I["🔑 키워드 밀도<br/>Economic Keywords"]
            J["📈 점수 분산<br/>Score Variance"]
            K["🎯 도메인 적합성<br/>Domain Relevance"]
            
            H --> L["⚖️ 종합 임계값<br/>Adjusted Threshold"]
            I --> L
            J --> L
            K --> L
        end
    end
### 🔧 **Augment (증강) 단계**
graph LR
    subgraph "2️⃣ AUGMENT 단계"
        A["📊 검색 결과<br/>Raw Documents"] --> B["🔄 리랭킹<br/>BGE-reranker-v2-m3-ko"]
        
        B --> C["📋 TOP_K=5<br/>Best Documents"]
        
        C --> D["🧠 고급 키워드 추출"]
        
        D --> E["📝 컨텍스트 구성<br/>Context Assembly"]
        
        E --> F["🎯 관련 질문 검색<br/>Related Questions"]
        
        F --> G["✅ 증강된 컨텍스트<br/>Augmented Context"]
        
        subgraph "🧠 키워드 추출 프로세스"
            H["📝 형태소 분석<br/>KiwiPiepy<br/>명사/동사/형용사"]
            I["🎯 의미적 확장<br/>Semantic Expansion<br/>Seed Keywords"]
            J["📈 TF-IDF 분석<br/>Term Importance"]
            K["🔗 키워드 통합<br/>Final Keywords"]
            
            H --> K
            I --> K
            J --> K
        end
        
        subgraph "📊 리랭킹 세부 과정"
            L["🔍 쿼리-문서 페어링<br/>Query-Doc Pairs"]
            M["🧮 교차 인코딩<br/>Cross-Encoding"]
            N["📊 관련성 점수<br/>Relevance Scores"]
            O["🏆 순위 재조정<br/>Re-ranking"]
            
            L --> M
            M --> N
            N --> O
        end
    end
### 🤖 **Generate (생성) 단계**
graph LR
    subgraph "3️⃣ GENERATE 단계"
        A["📝 증강된 컨텍스트<br/>Augmented Context"] --> B["📋 프롬프트 구성<br/>Prompt Engineering"]
        
        B --> C["🤖 LLM 추론<br/>Gemma-3-12b-it<br/>Multi-GPU"]
        
        C --> D["📊 신뢰도 계산<br/>Confidence Scoring"]
        
        D --> E["📚 출처 정보 첨부<br/>Source Attribution"]
        
        E --> F["✅ 최종 답변<br/>Final Answer"]
        
        subgraph "📋 프롬프트 구성"
            G["🎯 질문 분석<br/>Query Analysis"]
            H["📄 컨텍스트 정리<br/>Context Summary"]
            I["📚 출처 정보<br/>Source Information"]
            J["🔧 답변 형식 지정<br/>Answer Format"]
            
            G --> K["📝 완성된 프롬프트<br/>Complete Prompt"]
            H --> K
            I --> K
            J --> K
        end
        
        subgraph "📊 신뢰도 계산"
            L["🔍 벡터 유사도<br/>Vector Similarity<br/>(60% 가중치)"]
            M["🔄 리랭킹 점수<br/>Reranking Score<br/>(40% 가중치)"]
            N["📈 가중 평균<br/>Weighted Average"]
            O["🎯 최종 신뢰도<br/>Final Confidence<br/>(0-1 범위)"]
            
            L --> N
            M --> N
            N --> O
        end
    end

## 4. 시스템 아키텍처 다이어그램 🏗️
graph TB
    subgraph "🌐 프론트엔드 (Streamlit)"
        A["👤 사용자 인터페이스<br/>Chat Interface<br/>Port: 8501"]
    end
    
    subgraph "⚡ 백엔드 (FastAPI)"
        B["🔌 API 서버<br/>FastAPI<br/>Port: 8000"]
        C["🎯 엔드포인트<br/>/query, /search<br/>/health, /stats"]
    end
    
    subgraph "🧠 RAG 시스템 (rag_utils.py)"
        D["🔍 RAGSystem<br/>Main Controller"]
        E["📏 EmbeddingModel<br/>BGE-m3-ko"]
        F["🔄 RerankerModel<br/>BGE-reranker-v2-m3-ko"]
        G["🤖 AnswerGenerator<br/>Gemma-3-12b-it"]
        H["🗃️ VectorStore<br/>Qdrant Client"]
        I["📚 DocumentLoader<br/>Markdown Parser"]
        J["🔑 AdvancedKeywordExtractor<br/>KiwiPiepy + TF-IDF"]
    end
    
    subgraph "🗄️ 데이터 저장소"
        K["💾 Qdrant Vector DB<br/>Port: 6333<br/>Collection: multimodal_rag"]
        L["📄 Text Analysis<br/>llm_analysis_output/"]
        M["🖼️ Image Analysis<br/>analysis_output/"]
    end
    
    subgraph "📊 설정 및 로깅"
        N["⚙️ Config<br/>config.py"]
        O["📋 Logs<br/>logs/ directory"]
    end
    
    A <--> B
    B --> C
    C --> D
    
    D --> E
    D --> F
    D --> G
    D --> H
    D --> I
    D --> J
    
    H <--> K
    I --> L
    I --> M
    
    D --> N
    D --> O
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style D fill:#fff3e0
    style K fill:#e8f5e8

## 5. 주요 특징 및 혁신 사항 ⭐

### 🚀 **시스템의 핵심 혁신 기능들**

#### 1. **동적 임계값 시스템** 🎯
```
# 기존: 고정 임계값 (0.45)
# 개선: 질문 특성에 따른 동적 조정
- 질문 길이: 짧으면 엄격, 길면 관대
- 키워드 밀도: 경제 키워드 많으면 관대
- 점수 분산: 일관성 없으면 엄격
```

#### 2. **고급 키워드 추출** 🔑
```
# 3단계 키워드 추출
1. 형태소 분석 (KiwiPiepy): 명사/동사/형용사
2. 의미적 확장: 시드 키워드 기반 유사도 매칭
3. TF-IDF 분석: 문서 중요도 기반 가중치
```

#### 3. **멀티모달 데이터 처리** 🖼️
```
# 데이터 타입별 처리
- Text Analysis: LLM 분석 결과 (markdown_text_result/)
- Image Analysis: VLM 분석 결과 (markdown_results/)
- 섹션 타입: analysis, questions, image_summary
```

#### 4. **비관련 질문 유도 시스템** ⚠️
```
# 문서 연관성 실패 시
→ 한국은행 뉴스 관련 질문 유도
→ 구체적인 예시 질문 제공
→ 시스템 범위 안내
```

## 6. 성능 최적화 및 안정성 🔧

### 📊 **성능 지표**
- **임베딩 모델**: BGE-m3-ko (768차원)
- **검색 성능**: TOP_K=10 → 리랭킹 TOP_K=5
- **신뢰도**: 벡터 유사도(60%) + 리랭킹 점수(40%)
- **GPU 지원**: 멀티 GPU 병렬 처리

### 🛡️ **안정성 기능**
- **GPU 메모리 관리**: 자동 정리 및 모니터링
- **에러 핸들링**: 각 단계별 예외 처리
- **로깅 시스템**: 성능/오류/디버그 분리 로그
- **모델 상태 검증**: 텐서 건강성 체크

## 📝 **요약**


1. **🎯 Retrieve**: 동적 임계값과 키워드 보정을 통한 지능적 문서 검색
2. **🔧 Augment**: 리랭킹과 고급 키워드 추출을 통한 컨텍스트 증강
3. **🤖 Generate**: Gemma-3-12b-it 기반의 출처 명시형 답변 생성

**핵심 혁신**:
- 질문 특성에 따른 적응형 임계값 조정
- 형태소 분석 + 의미적 확장 + TF-IDF 통합 키워드 추출
- 텍스트와 이미지 분석 결과의 멀티모달 통합
- 비관련 질문에 대한 건설적 유도 답변 시스템