# Dockerfile
FROM python:3.10-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 & 필수 라이브러리 설치
RUN pip install --upgrade pip && \
    pip install "transformers>=4.51.0" torch accelerate fastapi uvicorn

# 사용자 설정
WORKDIR /app
COPY run.py .

# 포트 설정
EXPOSE 8000

# FastAPI 서버 실행
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]