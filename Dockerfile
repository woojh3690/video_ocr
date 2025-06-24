FROM python:3.11-slim
LABEL maintainer="woojh3690@iwaz.co.kr"

# 작업 디렉토리 설정
WORKDIR /app

# tzdata timezone 설정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg            \
        python3-opencv    \
        libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# python3-opencv 설치 경로 추가
ENV PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH

COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# 현재 디렉토리의 중요 파일을 컨테이너의 작업 디렉토리로 복사
COPY src/ /app/

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7340"]