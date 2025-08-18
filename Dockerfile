FROM python:3.11-slim
LABEL maintainer="woojh3690@gmail.com"

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

# python3-opencv 를 파이썬 검색 경로에 영구 추가
RUN python - <<'PY'
import site, pathlib
p = pathlib.Path(site.getsitepackages()[0]) / 'debian-dist-packages.pth'
p.write_text('/usr/lib/python3/dist-packages\n')
print('Added', p, '-> /usr/lib/python3/dist-packages')
PY

# 환경변수로도 dist-packages 를 노출
ENV PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH

COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# 현재 디렉토리의 중요 파일을 컨테이너의 작업 디렉토리로 복사
COPY src/ /app/

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7340"]
