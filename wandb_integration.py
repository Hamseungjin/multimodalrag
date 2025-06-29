"""
W&B 통합 및 플라이휠 워크플로우 모니터링
실험 추적, 메트릭 로깅, 모델 성능 분석을 위한 통합 시스템
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
from loguru import logger
import numpy as np

# W&B 설치 확인 및 import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("W&B가 설치되지 않았습니다. 메트릭 로깅이 로컬 파일로만 저장됩니다.")

class WandBFlyWheelTracker:
    """플라이휠 워크플로우를 위한 W&B 통합 추적기"""
    
    def __init__(self, project_name: str = "multimodal-rag-flywheel", 
                 entity: Optional[str] = None):
        self.project_name = project_name
        self.entity = entity
        self.wandb_run = None
        self.local_metrics = []  # W&B 없을 때 로컬 저장용
        
        # W&B 초기화
        if WANDB_AVAILABLE:
            self._initialize_wandb()
        else:
            logger.info("W&B 로컬 모드로 동작합니다.")
    
    def _initialize_wandb(self):
        """W&B 초기화"""
        try:
            # W&B 설정
            config = {
                "model_embedding": "dragonkue/BGE-m3-ko",
                "model_reranker": "dragonkue/bge-reranker-v2-m3-ko",
                "model_llm": "google/gemma-3-12b-it",
                "similarity_threshold": 0.45,
                "top_k_retrieval": 10,
                "top_k_rerank": 5,
                "adaptive_components": True,
                "flywheel_version": "1.0"
            }
            
            self.wandb_run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=config,
                tags=["flywheel", "adaptive-rag", "multimodal"],
                notes="플라이휠 워크플로우 기반 적응형 RAG 시스템"
            )
            
            logger.info(f"W&B 초기화 완료: {self.wandb_run.url}")
            
        except Exception as e:
            logger.error(f"W&B 초기화 실패: {e}")
            self.wandb_run = None
    
    def log_query_performance(self, query: str, result: Dict[str, Any], 
                            processing_time: float = None):
        """쿼리 성능 메트릭 로깅"""
        metrics = {
            "query_length": len(query),
            "confidence": result.get("confidence", 0.0),
            "sources_count": len(result.get("sources", [])),
            "answer_length": len(result.get("answer", "")),
            "domain": result.get("domain", "unknown"),
            "keywords_count": len(result.get("keywords", [])),
            "vector_weight": result.get("adaptive_weights", {}).get("vector", 0.6),
            "rerank_weight": result.get("adaptive_weights", {}).get("rerank", 0.4),
            "smart_threshold": result.get("smart_threshold", 0.45),
            "processing_time": processing_time or 0.0,
            "timestamp": time.time()
        }
        
        # W&B 로깅
        if self.wandb_run:
            wandb.log(metrics)
        
        # 로컬 저장
        self.local_metrics.append({
            "query": query[:100],  # 쿼리 일부만 저장
            "metrics": metrics
        })
        
        # 로컬 메트릭 크기 제한
        if len(self.local_metrics) > 1000:
            self.local_metrics = self.local_metrics[-800:]
    
    def log_keyword_performance(self, query: str, keyword_result: Dict[str, Any]):
        """키워드 추출 성능 로깅"""
        keywords_metrics = {
            "keyword_extraction/domain": keyword_result.get("domain", "unknown"),
            "keyword_extraction/basic_count": len(keyword_result.get("basic_keywords", [])),
            "keyword_extraction/semantic_count": len(keyword_result.get("semantic_keywords", [])),
            "keyword_extraction/tfidf_count": len(keyword_result.get("tfidf_keywords", [])),
            "keyword_extraction/final_count": len(keyword_result.get("final_keywords", [])),
            "keyword_extraction/query_length": len(query)
        }
        
        if self.wandb_run:
            wandb.log(keywords_metrics)
    
    def log_weight_adaptation(self, query: str, domain: str, vector_weight: float, 
                            rerank_weight: float, reasoning: str = ""):
        """가중치 적응 로깅"""
        weight_metrics = {
            "weight_adaptation/domain": domain,
            "weight_adaptation/vector_weight": vector_weight,
            "weight_adaptation/rerank_weight": rerank_weight,
            "weight_adaptation/query_length": len(query),
            "weight_adaptation/reasoning": reasoning
        }
        
        if self.wandb_run:
            wandb.log(weight_metrics)
    
    def log_threshold_adaptation(self, query: str, domain: str, smart_threshold: float, 
                               base_threshold: float = 0.45):
        """임계값 적응 로깅"""
        threshold_metrics = {
            "threshold_adaptation/domain": domain,
            "threshold_adaptation/smart_threshold": smart_threshold,
            "threshold_adaptation/base_threshold": base_threshold,
            "threshold_adaptation/adjustment": smart_threshold - base_threshold,
            "threshold_adaptation/query_length": len(query)
        }
        
        if self.wandb_run:
            wandb.log(threshold_metrics)
    
    def log_system_performance(self, performance_summary: Dict[str, Any]):
        """전체 시스템 성능 요약 로깅"""
        system_metrics = {
            "system/total_queries": performance_summary.get("total_queries", 0),
            "system/avg_confidence": performance_summary.get("recent_avg_confidence", 0.0),
            "system/avg_sources": performance_summary.get("recent_avg_sources", 0.0),
            "system/performance_trend": performance_summary.get("performance_trend", "unknown")
        }
        
        if self.wandb_run:
            wandb.log(system_metrics)
    
    def log_flywheel_iteration(self, iteration: int, improvements: Dict[str, Any]):
        """플라이휠 반복 개선사항 로깅"""
        flywheel_metrics = {
            "flywheel/iteration": iteration,
            "flywheel/keyword_accuracy_improvement": improvements.get("keyword_accuracy", 0.0),
            "flywheel/weight_optimization_improvement": improvements.get("weight_optimization", 0.0),
            "flywheel/threshold_precision_improvement": improvements.get("threshold_precision", 0.0),
            "flywheel/overall_performance_improvement": improvements.get("overall_performance", 0.0)
        }
        
        if self.wandb_run:
            wandb.log(flywheel_metrics)
    
    def create_performance_dashboard(self):
        """성능 대시보드 생성"""
        if not self.wandb_run:
            return None
        
        # 커스텀 차트 정의
        wandb.define_metric("confidence", summary="mean")
        wandb.define_metric("processing_time", summary="mean")
        wandb.define_metric("sources_count", summary="mean")
        
        # 도메인별 성능 차트
        wandb.define_metric("domain")
        wandb.define_metric("confidence", step_metric="domain")
        
        # 적응형 컴포넌트 성능 차트
        wandb.define_metric("weight_adaptation/*")
        wandb.define_metric("threshold_adaptation/*")
        wandb.define_metric("keyword_extraction/*")
        
        logger.info("W&B 성능 대시보드가 생성되었습니다.")
    
    def save_local_metrics(self, filepath: str = "local_metrics.json"):
        """로컬 메트릭을 파일로 저장"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.local_metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"로컬 메트릭이 {filepath}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"로컬 메트릭 저장 실패: {e}")
    
    def finish(self):
        """W&B 세션 종료"""
        if self.wandb_run:
            wandb.finish()
        
        # 로컬 메트릭 저장
        self.save_local_metrics()

class FlyWheelDataGenerator:
    """플라이휠 워크플로우를 위한 합성 데이터 생성기"""
    
    def __init__(self, wandb_tracker: WandBFlyWheelTracker):
        self.wandb_tracker = wandb_tracker
        self.generated_qa_pairs = []
    
    def generate_synthetic_qa_pairs(self, domain: str, count: int = 100) -> List[Dict[str, str]]:
        """도메인별 합성 Q&A 쌍 생성"""
        # 실제 구현에서는 GPT-4 API를 사용하여 고품질 Q&A 생성
        # 여기서는 예시 템플릿 제공
        
        economic_templates = [
            ("미국의 {indicator} 동향은 어떻게 되나요?", ["소비자물가", "고용지표", "GDP 성장률"]),
            ("{period} {country} 경제 상황을 분석해주세요.", ["2024년 1분기", "2023년"], ["미국", "한국", "일본"]),
            ("{policy}가 경제에 미치는 영향은?", ["금리 인상", "양적완화", "재정정책"])
        ]
        
        qa_pairs = []
        
        if domain == "economics":
            templates = economic_templates
        else:
            templates = [("일반적인 질문입니다.", [])]
        
        for i in range(min(count, len(templates) * 10)):
            template, options = templates[i % len(templates)][:2]
            if len(templates[i % len(templates)]) > 2:
                # 복합 템플릿 처리
                pass
            
            # 간단한 템플릿 기반 생성 (실제로는 GPT-4 사용)
            question = template.format(
                indicator=options[i % len(options)] if options else "지표",
                period="2024년",
                country="미국",
                policy="통화정책"
            )
            
            answer = f"이 질문에 대한 분석 답변입니다: {question}"
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "domain": domain,
                "generated_at": time.time()
            })
        
        self.generated_qa_pairs.extend(qa_pairs)
        
        # W&B에 생성 통계 로깅
        self.wandb_tracker.log_system_performance({
            "synthetic_qa_generated": len(qa_pairs),
            "domain": domain,
            "total_synthetic_qa": len(self.generated_qa_pairs)
        })
        
        return qa_pairs
    
    def evaluate_qa_quality(self, qa_pairs: List[Dict[str, str]]) -> Dict[str, float]:
        """생성된 Q&A 품질 평가"""
        # 간단한 품질 메트릭 계산
        quality_metrics = {
            "avg_question_length": np.mean([len(qa["question"]) for qa in qa_pairs]),
            "avg_answer_length": np.mean([len(qa["answer"]) for qa in qa_pairs]),
            "domain_consistency": 1.0,  # 실제로는 도메인 일치도 계산
            "question_diversity": len(set(qa["question"][:50] for qa in qa_pairs)) / len(qa_pairs)
        }
        
        return quality_metrics

class FlyWheelExperimentManager:
    """플라이휠 워크플로우 실험 관리자"""
    
    def __init__(self, wandb_tracker: WandBFlyWheelTracker):
        self.wandb_tracker = wandb_tracker
        self.experiments = []
        self.current_experiment = None
    
    def start_experiment(self, experiment_name: str, config: Dict[str, Any]):
        """새로운 실험 시작"""
        experiment = {
            "name": experiment_name,
            "config": config,
            "start_time": time.time(),
            "metrics": [],
            "results": {}
        }
        
        self.current_experiment = experiment
        self.experiments.append(experiment)
        
        # W&B에 실험 시작 로깅
        if self.wandb_tracker.wandb_run:
            wandb.config.update(config)
        
        logger.info(f"실험 시작: {experiment_name}")
        return experiment
    
    def log_experiment_result(self, metric_name: str, value: float, step: int = None):
        """실험 결과 로깅"""
        if not self.current_experiment:
            logger.warning("실행 중인 실험이 없습니다.")
            return
        
        metric_entry = {
            "metric": metric_name,
            "value": value,
            "step": step,
            "timestamp": time.time()
        }
        
        self.current_experiment["metrics"].append(metric_entry)
        
        # W&B 로깅
        if self.wandb_tracker.wandb_run:
            log_data = {metric_name: value}
            if step is not None:
                log_data["step"] = step
            wandb.log(log_data)
    
    def finish_experiment(self, final_results: Dict[str, Any]):
        """현재 실험 종료"""
        if not self.current_experiment:
            logger.warning("실행 중인 실험이 없습니다.")
            return
        
        self.current_experiment["end_time"] = time.time()
        self.current_experiment["duration"] = (
            self.current_experiment["end_time"] - self.current_experiment["start_time"]
        )
        self.current_experiment["results"] = final_results
        
        logger.info(f"실험 완료: {self.current_experiment['name']}")
        logger.info(f"결과: {final_results}")
        
        self.current_experiment = None
    
    def get_best_experiment(self, metric_name: str) -> Optional[Dict]:
        """특정 메트릭 기준 최고 성능 실험 반환"""
        if not self.experiments:
            return None
        
        best_experiment = None
        best_value = float('-inf')
        
        for experiment in self.experiments:
            if metric_name in experiment.get("results", {}):
                value = experiment["results"][metric_name]
                if value > best_value:
                    best_value = value
                    best_experiment = experiment
        
        return best_experiment

# 전역 W&B 트래커 인스턴스 (싱글톤 패턴)
_global_wandb_tracker = None

def get_global_wandb_tracker() -> WandBFlyWheelTracker:
    """전역 W&B 트래커 인스턴스 반환"""
    global _global_wandb_tracker
    if _global_wandb_tracker is None:
        _global_wandb_tracker = WandBFlyWheelTracker()
    return _global_wandb_tracker

def initialize_flywheel_tracking(project_name: str = "multimodal-rag-flywheel", 
                                entity: str = None):
    """플라이휠 워크플로우 추적 초기화"""
    global _global_wandb_tracker
    _global_wandb_tracker = WandBFlyWheelTracker(project_name, entity)
    _global_wandb_tracker.create_performance_dashboard()
    return _global_wandb_tracker 