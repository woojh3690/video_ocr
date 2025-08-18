FROM python:3.12-slim
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
# /usr/lib/python3*/dist-packages 를 전부 모아 기록 (3.11/3.12 등 어떤 버전이든 커버)
RUN python - <<'PY'
import site, pathlib, glob, os
site_dir = pathlib.Path(site.getsitepackages()[0])
site_dir.mkdir(parents=True, exist_ok=True)
pth = site_dir / 'debian-dist-packages.pth'
candidates = sorted(set(glob.glob('/usr/lib/python3*/dist-packages')))
candidates = [p for p in candidates if os.path.isdir(p)]
pth.write_text('\n'.join(candidates) + '\n')
print("Wrote", pth, "with:")
print(pth.read_text())
PY

# 환경변수로도 dist-packages 를 노출
ENV PYTHONPATH=/usr/lib/python3/dist-packages:/usr/lib/python3.11/dist-packages${PYTHONPATH:+:$PYTHONPATH}

# opencv import 검증
RUN python - <<'PY'
import sys, site
print("sys.version=", sys.version)
print("site.getsitepackages()=", site.getsitepackages())
print("sys.path has dist-packages ->", [p for p in sys.path if "dist-packages" in p])
import cv2
print("cv2.__version__=", cv2.__version__)
print("cv2.__file__=", cv2.__file__)
PY

COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# 현재 디렉토리의 중요 파일을 컨테이너의 작업 디렉토리로 복사
COPY src/ /app/

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7340"]
