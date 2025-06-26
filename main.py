"""
MultiModal RAG 시스템 FastAPI 백엔드
"""
import os
import sys
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from config import Config
from rag_utils import RAGSystem

# 전역 RAG 시스템 변수
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 앱 생명주기 관리"""
    global rag_system
    
    # 시작 시 RAG 시스템 초기화
    logger.info("RAG 시스템 초기화 중...")
    try:
        rag_system = RAGSystem()
        logger.info("RAG 시스템 초기화 완료")
    except Exception as e:
        logger.error(f"RAG 시스템 초기화 실패: {e}")
        raise
    
    yield
    
    # 종료 시 정리 작업
    logger.info("FastAPI 서버 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="MultiModal RAG API",
    description="한국은행 뉴스 데이터를 활용한 MultiModal RAG 시스템 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델 정의
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str
    query_analysis: Optional[str] = ""
    related_questions: Optional[List[str]] = []

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str

class HealthResponse(BaseModel):
    status: str
    message: str

def setup_logging():
    """로깅을 설정합니다."""
    Config.create_directories()
    
    logger.remove()
    
    # 콘솔 로거 설정
    logger.add(
        sys.stdout,
        level=Config.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 일반 로그 파일
    logger.add(
        Config.LOG_FILE,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="50 MB",
        retention="10 days",
        encoding="utf-8"
    )
    
    # 오류 전용 로그 파일
    logger.add(
        Config.LOG_ERROR_FILE,
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8"
    )
    
    # 성능 로그 파일 (DEBUG 레벨)
    logger.add(
        Config.LOG_PERFORMANCE_FILE,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="20 MB",
        retention="7 days",
        encoding="utf-8",
        filter=lambda record: "소요시간" in record["message"] or "메모리" in record["message"] or "GPU" in record["message"]
    )

@app.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트"""
    return HealthResponse(
        status="success",
        message="MultiModal RAG API가 정상적으로 실행 중입니다."
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    global rag_system
    
    if rag_system is None:
        return HealthResponse(
            status="error",
            message="RAG 시스템이 초기화되지 않았습니다."
        )
    
    return HealthResponse(
        status="success",
        message="RAG 시스템이 정상적으로 작동 중입니다."
    )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """문서 검색 엔드포인트"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    try:
        # 쿼리 임베딩
        query_embedding = rag_system.embedding_model.encode([request.query])
        
        # 검색 수행
        search_results = rag_system.vector_store.search(
            query_embedding[0],
            top_k=request.top_k
        )
        
        return SearchResponse(
            results=search_results,
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def answer_query(request: QueryRequest):
    """질의응답 엔드포인트"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    try:
        logger.info(f"질문 처리 시작: {request.query}")
        
        # RAG 시스템을 통한 검색 및 답변 생성
        result = rag_system.search_and_answer(
            query=request.query,
            top_k=request.top_k
        )
        
        logger.info(f"질문 처리 완료: {request.query}")
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            query=request.query,
            query_analysis=result.get("query_analysis", ""),
            related_questions=result.get("related_questions", [])
        )
        
    except Exception as e:
        logger.error(f"질의응답 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"질의응답 중 오류가 발생했습니다: {str(e)}")

@app.get("/memory")
async def get_memory_status():
    """멀티 GPU 메모리 상태 조회"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"status": "CPU 모드", "gpu_available": False}
        
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        total_memory = 0
        total_allocated = 0
        total_cached = 0
        
        # 모든 GPU 정보 수집
        for i in range(gpu_count):
            device_props = torch.cuda.get_device_properties(i)
            device_memory = device_props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free_memory = device_memory - allocated
            
            gpu_info.append({
                "gpu_id": i,
                "device_name": device_props.name,
                "total_memory_gb": round(device_memory, 2),
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
                "free_memory_gb": round(free_memory, 2),
                "memory_usage_percent": round((allocated / device_memory) * 100, 1)
            })
            
            total_memory += device_memory
            total_allocated += allocated
            total_cached += cached
        
        return {
            "status": f"멀티 GPU 사용중 (x{gpu_count})",
            "gpu_available": True,
            "gpu_count": gpu_count,
            "total_memory_gb": round(total_memory, 2),
            "total_allocated_gb": round(total_allocated, 2),
            "total_cached_gb": round(total_cached, 2),
            "total_free_memory_gb": round(total_memory - total_allocated, 2),
            "average_usage_percent": round((total_allocated / total_memory) * 100, 1),
            "gpus": gpu_info
        }
        
    except Exception as e:
        logger.error(f"메모리 상태 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"메모리 상태 조회 실패: {str(e)}")

@app.get("/stats")
async def get_stats():
    """시스템 통계 정보 조회"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다.")
    
    try:
        # Qdrant 컬렉션 정보 조회
        collection_info = rag_system.vector_store.client.get_collection(Config.COLLECTION_NAME)
        
        # 안전한 값 추출
        vector_size = getattr(collection_info.config.params.vectors, 'size', 1024)
        distance_metric = getattr(collection_info.config.params.vectors.distance, 'value', 'COSINE')
        
        return {
            "collection_name": Config.COLLECTION_NAME,
            "total_documents": collection_info.points_count or 0,
            "vector_size": vector_size,
            "distance_metric": distance_metric,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.LLM_MODEL
        }
        
    except Exception as e:
        logger.error(f"통계 조회 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    setup_logging()
    
    import uvicorn
    
    logger.info("FastAPI 서버 시작")
    uvicorn.run(
        "main:app",
        host=Config.FASTAPI_HOST,
        port=Config.FASTAPI_PORT,
        reload=False,
        log_level="info"
    ) 