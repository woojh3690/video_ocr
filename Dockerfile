FROM debian:trixie-slim
LABEL maintainer="woojh3690@gmail.com"

# 작업 디렉토리
WORKDIR /app

# 비대화형 tzdata, 타임존
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 필수 패키지 설치 (브리지 불필요)
# - python3/python3-pip: 런타임/패키지 설치
# - python3-opencv: 시스템 FFmpeg과 링크된 OpenCV 파이썬 바인딩
# - ffmpeg: AV1 등 디코딩/인코딩
# - libgl1, libglib2.0-0, libsm6, libxext6: OpenCV 런타임 의존성(HIGHGUI/VideoIO 등)
# - tzdata: 타임존 설정
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-opencv \
        ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 \
        tzdata; \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime; echo $TZ > /etc/timezone; \
    rm -rf /var/lib/apt/lists/*

# venv 생성(+ 시스템 site-packages 사용하여 cv2 바로 보이게)
RUN python3 -m venv --system-site-packages /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip wheel setuptools

# 런타임이 venv를 기본으로 쓰도록 PATH 설정
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# (옵션) 빌드 타임 검증: venv 파이썬에서 cv2 임포트 확인
RUN python - <<'PY'
import sys, cv2
print("sys.version=", sys.version)
print("cv2.__version__=", cv2.__version__)
print("cv2.__file__=", cv2.__file__)
PY

# 파이썬 의존성 설치 (캐시 효율을 위해 requirements 먼저 복사)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY src/ /app/

# 실행 (venv의 uvicorn 사용)
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7340"]
