FROM python:3.11
LABEL maintainer="woojh3690@iwaz.co.kr"

# 작업 디렉토리 설정
WORKDIR /app

# tzdata timezone 설정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 의존성 파일 복사 및 설치
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# 현재 디렉토리의 중요 파일을 컨테이너의 작업 디렉토리로 복사
COPY src/ /app/

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7340"]