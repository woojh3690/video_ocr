FROM python:3.12-slim
LABEL maintainer="woojh3690@gmail.com"

# 작업 디렉토리 설정
WORKDIR /app

# tzdata timezone 설정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN set -eux; \
    echo "APT_CACHE_BUST=${APT_CACHE_BUST}"; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        python3-opencv \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        tzdata && \
    rm -rf /var/lib/apt/lists/*

# python3-opencv 를 파이썬 검색 경로에 영구 추가
# /usr/lib/python3*/dist-packages 를 전부 모아 기록 (3.11/3.12 등 어떤 버전이든 커버)
RUN python - <<'PY'
import site, pathlib, glob, os
site_dir = pathlib.Path(site.getsitepackages()[0])
site_dir.mkdir(parents=True, exist_ok=True)
pth = site_dir / 'debian-dist-packages.pth'
candidates = sorted(set(p for p in glob.glob('/usr/lib/python3*/dist-packages') if os.path.isdir(p)))
pth.write_text('\n'.join(candidates) + '\n')
print("Wrote", pth, "with:\n" + pth.read_text())
PY

# opencv import 검증 + 실제 설치 파일 경로 확인
RUN bash -lc "dpkg -L python3-opencv | grep -E '/cv2($|/)' || true"
RUN python - <<'PY'
import sys, site, os
print("sys.version=", sys.version)
print("site.getsitepackages()=", site.getsitepackages())
print("sys.path (dist-packages) ->", [p for p in sys.path if "dist-packages" in p])
print("cv2 dir exists?",
      any(os.path.exists(os.path.join(p, 'cv2')) for p in sys.path if 'dist-packages' in p))
import cv2
print("cv2.__version__=", cv2.__version__)
print("cv2.__file__=", cv2.__file__)
PY

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY src/ /app/

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7340"]
