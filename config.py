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
    
    # 로깅 설정 (개선)
    LOG_LEVEL = "DEBUG"  # 더 상세한 로깅
    LOG_FILE = PROJECT_ROOT / "logs" / "rag_system.log"
    LOG_ERROR_FILE = PROJECT_ROOT / "logs" / "rag_errors.log"
    LOG_PERFORMANCE_FILE = PROJECT_ROOT / "logs" / "performance.log"
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리들을 생성합니다."""
        (cls.PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        # 로그 파일들 초기화
        for log_file in [cls.LOG_FILE, cls.LOG_ERROR_FILE, cls.LOG_PERFORMANCE_FILE]:
            log_file.parent.mkdir(exist_ok=True)
            if not log_file.exists():
                log_file.touch() 