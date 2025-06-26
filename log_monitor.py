#!/usr/bin/env python3
"""
MultiModalRAG 로그 모니터링 유틸리티
"""
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import argparse

def monitor_logs(log_dir: Path = Path("logs"), watch_interval: int = 5):
    """로그 파일들을 실시간으로 모니터링합니다."""
    print(f"🔍 로그 모니터링 시작: {log_dir}")
    print(f"📊 모니터링 간격: {watch_interval}초")
    print("-" * 50)
    
    last_positions = {}
    
    while True:
        try:
            for log_file in log_dir.glob("*.log"):
                if log_file.is_file():
                    current_size = log_file.stat().st_size
                    last_pos = last_positions.get(str(log_file), 0)
                    
                    if current_size > last_pos:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            f.seek(last_pos)
                            new_lines = f.read()
                            
                            if new_lines.strip():
                                print(f"\n📋 {log_file.name}:")
                                print(new_lines.strip())
                                print("-" * 30)
                        
                        last_positions[str(log_file)] = current_size
            
            time.sleep(watch_interval)
            
        except KeyboardInterrupt:
            print("\n⏹️ 모니터링 중단")
            break
        except Exception as e:
            print(f"❌ 모니터링 오류: {e}")
            time.sleep(watch_interval)

def analyze_errors(log_dir: Path = Path("logs"), hours: int = 24):
    """최근 시간대의 오류 로그를 분석합니다."""
    print(f"🔍 최근 {hours}시간 오류 분석")
    print("-" * 50)
    
    error_patterns = {
        "probability tensor": "수치 불안정성 오류",
        "GPU 메모리": "GPU 메모리 부족",
        "생성 시도 실패": "모델 생성 실패",
        "토큰화 중 오류": "토큰화 문제",
        "답변이 너무 짧음": "답변 품질 문제",
        "Half": "Float16 호환성 문제",
        "softmax_lastdim_kernel_impl": "Softmax Float16 오류",
        "로짓 통계에 NaN": "로짓 NaN 오류",
        "Float16 호환성 문제": "정밀도 문제"
    }
    
    error_counts = {pattern: 0 for pattern in error_patterns}
    recent_errors = []
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    for log_file in log_dir.glob("*.log"):
        if log_file.is_file():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 로그 시간 파싱
                        if " | " in line:
                            try:
                                timestamp_str = line.split(" | ")[0]
                                log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                                
                                if log_time >= cutoff_time:
                                    # 오류 패턴 검사
                                    for pattern, description in error_patterns.items():
                                        if pattern in line.lower():
                                            error_counts[pattern] += 1
                                            recent_errors.append({
                                                "time": timestamp_str,
                                                "file": log_file.name,
                                                "pattern": pattern,
                                                "description": description,
                                                "line": line.strip()
                                            })
                                            break
                            except ValueError:
                                continue
            except Exception as e:
                print(f"❌ {log_file.name} 분석 중 오류: {e}")
    
    # 결과 출력
    print("📊 오류 통계:")
    for pattern, count in error_counts.items():
        if count > 0:
            print(f"  • {error_patterns[pattern]}: {count}회")
    
    if recent_errors:
        print(f"\n📝 최근 오류 목록 (최대 10개):")
        for error in recent_errors[-10:]:
            print(f"  🕐 {error['time']} - {error['description']}")
            print(f"     📄 {error['file']}: {error['line'][:100]}...")
            print()
    else:
        print("✅ 최근 시간대에 주요 오류가 없습니다!")

def get_system_status():
    """시스템 상태를 요약합니다."""
    print("🖥️ 시스템 상태 체크")
    print("-" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🎮 GPU 상태:")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"  • GPU {i}: {device_name}")
                print(f"    메모리: {memory_allocated:.1f}GB / {total_memory:.1f}GB ({memory_allocated/total_memory*100:.1f}%)")
        else:
            print("❌ GPU 사용 불가능")
    except ImportError:
        print("❌ PyTorch 없음")
    
    # 로그 파일 상태
    log_dir = Path("logs")
    if log_dir.exists():
        print(f"\n📁 로그 파일 상태:")
        for log_file in log_dir.glob("*.log"):
            size_mb = log_file.stat().st_size / 1024**2
            modified = datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"  • {log_file.name}: {size_mb:.1f}MB (수정: {modified.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        print("❌ 로그 디렉토리 없음")

def main():
    parser = argparse.ArgumentParser(description="MultiModalRAG 로그 모니터링 도구")
    parser.add_argument("--monitor", action="store_true", help="실시간 로그 모니터링")
    parser.add_argument("--analyze", type=int, default=24, help="최근 N시간 오류 분석 (기본: 24)")
    parser.add_argument("--status", action="store_true", help="시스템 상태 확인")
    parser.add_argument("--log-dir", type=str, default="logs", help="로그 디렉토리 경로")
    
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    
    if args.monitor:
        monitor_logs(log_dir)
    elif args.analyze:
        analyze_errors(log_dir, args.analyze)
    elif args.status:
        get_system_status()
    else:
        print("🚀 MultiModalRAG 로그 분석 도구")
        print("=" * 50)
        get_system_status()
        print()
        analyze_errors(log_dir, 24)

if __name__ == "__main__":
    main() 