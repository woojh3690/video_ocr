# Repository Guidelines

## Project Structure & Module Organization
- `src/main.py`: FastAPI app (routes, websocket, task queue, video streaming).
- `src/core/`: OCR pipeline and helpers (`ocr.py`, `csv_to_srt.py`, `docker_manager.py`).
- `src/templates/` and `src/static/`: Jinja2 templates and assets.
- `src/uploads/`: Uploaded videos and outputs (served via `/videos/...`).
- `test/`: Test data (e.g., `test_image/`). No formal unit tests yet.
- `Dockerfile`: Production image (Python 3.11, ffmpeg, OpenCV, uvicorn entrypoint).

## Build, Test, and Development Commands
- Install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run API (from `src/`): `uvicorn main:app --reload --host 0.0.0.0 --port 7340`
- Docker build/run: `docker build -t video-ocr . && docker run -p 7340:7340 video-ocr`
- Quick health: `curl http://localhost:7340/` (serves index page).

## Coding Style & Naming Conventions
- Python 3.11; follow PEP 8 with 4‑space indentation and type hints where practical.
- Names: `snake_case` for functions/variables, `CamelCase` for classes, module filenames in `snake_case`.
- Keep FastAPI endpoints small; move OCR logic into `src/core/`.
- Prefer explicit env access (e.g., `DOCKER_URL`, `DOCKER_NAME`, `KAFKA_URL`) with safe defaults.

## Testing Guidelines
- Framework: pytest recommended. Place tests under `test/` with `test_*.py` naming.
- Start with route tests using `fastapi.testclient.TestClient` and unit tests for `core/ocr.py` helpers.
- Run: `pytest -q` (from repo root). Aim to cover error paths and large file handling.

## Commit & Pull Request Guidelines
- Commit style: use Korean type prefixes found in history — `추가:` (feature), `수정:` (fix/change), `리팩토링:`, `스타일:`, `삭제:`, `문서:`. Keep the subject concise (≤ 72 chars).
- Examples: `추가: 한번에 하나의 작업만 프로세싱하도록 제한`, `수정: Dockerfile maintainer 이메일 주소 업데이트`, `리팩토링: task queue 로직 정리`, `스타일: 파일명 칼럼 폭 200px -> 400px`, `삭제: 불필요한 테스트 스크립트 제거`.
- Language: Korean preferred (consistent with history); English acceptable when clearer.
- PRs: include problem/solution summary, linked issues, manual test steps, and UI screenshots when relevant. One topic per PR.
- Hygiene: pass local checks, avoid committing secrets or large media. Merge commits are fine (GitHub PRs); title should follow the same prefix style.

## Security & Configuration Tips
- Configure external services via env: `DOCKER_URL`, `DOCKER_NAME`, `KAFKA_URL`; avoid committing credentials.
- Video files go under `src/uploads/`; validate filenames and user input.
- The app exposes port `7340`; ensure reverse-proxy or firewall rules in production.
