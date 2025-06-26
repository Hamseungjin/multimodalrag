

---
## 🧭 전체 명령어 흐름 정리

---

### 🔍 1. GPU 상태 확인

```bash
nvidia-smi
python -c "import torch; print(f'GPU 개수: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

---

### 🧠 2. 멀티 GPU 환경 변수 설정

```bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
```

---

### 🛠️ 3. 가상환경 생성 및 의존성 설치

```bash
# uv 설치
pip install uv

# 가상환경 생성
uv venv .venv

# 가상환경 활성화
source .venv/bin/activate    # Linux/macOS
# 또는
.venv\Scripts\activate       # Windows

# 의존성 설치
uv pip install -r requirements.txt

# 추가 모델 관련 의존성
uv pip install accelerate
uv pip install flash-attn==2.6.3 --no-build-isolation
```

---

### 🧱 4. Qdrant 서버 설치 및 실행

```bash
# Qdrant 다운로드 및 실행
wget https://github.com/qdrant/qdrant/releases/download/v1.8.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
./qdrant --version

# 선택: Web UI 다운로드
wget https://github.com/qdrant/qdrant-web-ui/releases/download/v0.1.40/dist-qdrant.zip
unzip dist-qdrant.zip
mv dist static

# Qdrant 서버 실행
./qdrant --uri http://0.0.0.0:6333 &
```

---

### 🧬 5. 데이터 임베딩 실행

```bash
python embed_data.py
```

---

### 🚀 6. 시스템 실행

```bash
# 터미널 1 - FastAPI 백엔드
python main.py

# 터미널 2 - Streamlit 프론트엔드
streamlit run app.py

# 터미널 3 - 실시간 모니터링(gpu, 에러, 시스템 총 출력)
python log_monitor.py --monitor
```
