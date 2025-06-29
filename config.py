"""
MultiModal RAG 시스템 설정 파일
"""
import os
from pathlib import Path

class Config:
    # 프로젝트 경로 설정
    PROJECT_ROOT = Path(__file__).parent
    
    # 데이터 경로 설정
    LLM_ANALYSIS_PATH = PROJECT_ROOT / "llm_analysis_output"/ "markdown_text_result"
    IMAGE_ANALYSIS_PATH = PROJECT_ROOT / "analysis_output" / "markdown_results"
    
    # Qdrant 설정
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    COLLECTION_NAME = "multimodal_rag"
    
    # 임베딩 모델 설정
    EMBEDDING_MODEL = "dragonkue/BGE-m3-ko"
    RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"
    
    # LLM 설정 (답변 생성용)
    LLM_MODEL = "google/gemma-3-12b-it"  # 더 큰 Gemma 모델 사용

    
    # 검색 설정
    TOP_K_RETRIEVAL = 10
    TOP_K_RERANK = 5
    SIMILARITY_THRESHOLD = 0.45  # 문서 연관성 판단을 위한 임계값 (0.6에서 0.45로 하향)
    
    # 청크 설정
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # 서버 설정
    FASTAPI_HOST = "0.0.0.0"
    FASTAPI_PORT = 8000
    STREAMLIT_PORT = 8501
    
    # GPU 설정
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # 멀티 GPU 설정
    MULTI_GPU_STRATEGY = "model_parallel"  # "model_parallel" 또는 "data_parallel"
    GPU_MEMORY_FRACTION = 0.9  # GPU 메모리의 90% 사용
    ENABLE_GRADIENT_CHECKPOINTING = False  # 추론 시에는 비활성화
    
    # Gemma-3-12b-it 전용 설정
    FORCE_FLOAT32_FOR_GEMMA = True  # Gemma 모델의 Float16 호환성 문제 해결
    ENABLE_MIXED_PRECISION = False  # 혼합 정밀도 비활성화 (안정성 우선)
    
    # 적응형 컴포넌트 설정 (플라이휠 워크플로우)
    ENABLE_ADAPTIVE_COMPONENTS = True  # 적응형 컴포넌트 활성화
    KEYWORD_CACHE_SIZE = 1000  # 키워드 캐시 최대 크기
    KEYWORD_CACHE_TTL = 3600   # 키워드 캐시 TTL (초)
    WEIGHT_CACHE_TTL = 1800    # 가중치 캐시 TTL (초)
    
    # 도메인 감지 설정
    ENABLE_LLM_DOMAIN_DETECTION = True  # LLM 기반 도메인 감지 활성화
    DOMAIN_CACHE_TTL = 7200  # 도메인 캐시 TTL (2시간)
    DOMAIN_FALLBACK_TO_KEYWORD = True  # LLM 실패시 키워드 방식 fallback
    
    # Kiwi 형태소 분석기 설정
    ENABLE_KIWI_MORPHOLOGY = True  # Kiwi 형태소 분석기 사용
    KIWI_FALLBACK_TO_REGEX = True  # Kiwi 실패시 정규식 fallback
    
    # 도메인별 기본 가중치
    DOMAIN_WEIGHTS = {
        "economics": {"vector": 0.65, "rerank": 0.35},
        "finance": {"vector": 0.6, "rerank": 0.4},
        "general": {"vector": 0.6, "rerank": 0.4}
    }
    
    # 스마트 임계값 설정
    SMART_THRESHOLD_RANGE = (0.2, 0.7)  # (최소, 최대)
    ECONOMIC_DOMAIN_BOOST = -0.05  # 경제 도메인 임계값 조정
    
    # 플라이휠 워크플로우 설정
    FLYWHEEL_METRICS_SIZE = 1000  # 메트릭 히스토리 크기
    ENABLE_WANDB = False  # W&B 통합 (선택적)
    WANDB_PROJECT = "multimodal-rag-flywheel"
    
    # A/B 테스트 설정
    ENABLE_AB_TESTING = True
    AB_TEST_RATIO = 0.5  # 50%씩 분할
    AB_TEST_VARIANTS = {
        "A": {
            "name": "기존_가중치",
            "description": "고정 가중치 60-40 방식",
            "vector_weight": 0.6,
            "rerank_weight": 0.4,
            "threshold": 0.45,
            "use_adaptive": False
        },
        "B": {
            "name": "적응형_가중치", 
            "description": "도메인별 적응형 가중치",
            "use_adaptive_weights": True,
            "use_smart_threshold": True,
            "use_adaptive": True
        }
    }
    
    # 합성 데이터 생성 설정
    SYNTHETIC_QA_BATCH_SIZE = 50  # 한 번에 생성할 Q&A 쌍 수
    QUALITY_THRESHOLD = 0.7  # 합성 데이터 품질 임계값
    
    # 로깅 설정 (개선)
    LOG_LEVEL = "DEBUG"  # 더 상세한 로깅
    LOG_FILE = PROJECT_ROOT / "logs" / "rag_system.log"
    LOG_ERROR_FILE = PROJECT_ROOT / "logs" / "rag_errors.log"
    LOG_PERFORMANCE_FILE = PROJECT_ROOT / "logs" / "performance.log"
    LOG_FLYWHEEL_FILE = PROJECT_ROOT / "logs" / "flywheel_metrics.log"
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리들을 생성합니다."""
        (cls.PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        # 로그 파일들 초기화
        for log_file in [cls.LOG_FILE, cls.LOG_ERROR_FILE, cls.LOG_PERFORMANCE_FILE, cls.LOG_FLYWHEEL_FILE]:
            log_file.parent.mkdir(exist_ok=True)
            if not log_file.exists():
                log_file.touch() 