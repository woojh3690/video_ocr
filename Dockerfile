FROM debian:trixie-slim
LABEL maintainer="woojh3690@gmail.com"

WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# python3.13 계열을 명시적으로 설치 + OpenCV는 시스템 패키지
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3-pip \
        python3-opencv \
        ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 \
        tzdata; \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime; echo $TZ > /etc/timezone; \
    rm -rf /var/lib/apt/lists/*

# 3.13로 venv 생성 (+ 시스템 site-packages 접근: apt의 cv2 사용)
RUN /usr/bin/python3.13 -m venv --system-site-packages /opt/venv && \
    /opt/venv/bin/python -m ensurepip --upgrade && \
    /opt/venv/bin/python -m pip install --upgrade pip wheel setuptools

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# 빌드 타임 검증
RUN python - <<'PY'
import sys, cv2
print("sys.version=", sys.version)
print("cv2.__version__=", cv2.__version__)
print("cv2.__file__=", cv2.__file__)
PY

# 의존성/소스
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY src/ /app/

# 실행 (venv의 uvicorn 사용)
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7340"]
