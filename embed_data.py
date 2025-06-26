"""
데이터 임베딩 및 벡터 스토어 저장 스크립트
"""
import os
import sys
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from config import Config
from rag_utils import DocumentLoader, EmbeddingModel, VectorStore

def setup_logging():
    """로깅을 설정합니다."""
    Config.create_directories()
    
    logger.remove()
    logger.add(
        sys.stdout,
        level=Config.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        Config.LOG_FILE,
        level=Config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        rotation="100 MB"
    )

def load_all_documents():
    """모든 마크다운 문서를 로드합니다."""
    logger.info("문서 로딩 시작")
    
    all_documents = []
    
    # 1. 텍스트 분석 결과 로드
    text_docs = DocumentLoader.load_markdown_files(
        Config.LLM_ANALYSIS_PATH, 
        doc_type="text"
    )
    all_documents.extend(text_docs)
    logger.info(f"텍스트 분석 문서 {len(text_docs)}개 로드 완료")
    
    # 2. 이미지 분석 결과 로드
    image_docs = DocumentLoader.load_markdown_files(
        Config.IMAGE_ANALYSIS_PATH, 
        doc_type="image"
    )
    all_documents.extend(image_docs)
    logger.info(f"이미지 분석 문서 {len(image_docs)}개 로드 완료")
    
    logger.info(f"총 {len(all_documents)}개 문서 로드 완료")
    return all_documents

def embed_and_store_documents(documents):
    """문서들을 임베딩하고 벡터 스토어에 저장합니다."""
    if not documents:
        logger.warning("임베딩할 문서가 없습니다.")
        return
    
    logger.info("임베딩 모델 초기화")
    embedding_model = EmbeddingModel()
    
    logger.info("벡터 스토어 초기화")
    vector_store = VectorStore()
    
    # 문서 내용 추출
    texts = [doc.content for doc in documents]
    
    logger.info(f"{len(texts)}개 문서 임베딩 시작")
    
    # 배치 단위로 임베딩 생성
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="임베딩 생성"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch_texts, batch_size=len(batch_texts))
        all_embeddings.extend(batch_embeddings)
    
    logger.info("임베딩 생성 완료")
    
    # 벡터 스토어에 저장
    logger.info("벡터 스토어에 저장 시작")
    import numpy as np
    embeddings_array = np.array(all_embeddings)
    vector_store.add_documents(documents, embeddings_array)
    
    logger.info("벡터 스토어 저장 완료")

def main():
    """메인 함수"""
    try:
        setup_logging()
        logger.info("=== 데이터 임베딩 프로세스 시작 ===")
        
        # 1. 문서 로드
        documents = load_all_documents()
        
        if not documents:
            logger.error("로드된 문서가 없습니다. 프로세스를 종료합니다.")
            return
        
        # 2. 임베딩 및 저장
        embed_and_store_documents(documents)
        
        logger.info("=== 데이터 임베딩 프로세스 완료 ===")
        
    except Exception as e:
        logger.error(f"프로세스 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 