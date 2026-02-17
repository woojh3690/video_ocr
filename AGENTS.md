# Repository Guidelines

## Project Structure & Module Organization
- `src/main.py`: FastAPI app entrypoint (routes, websocket, queue, streaming).
- `src/core/`: Core OCR/business modules (`ocr.py`, `jsonl_to_srt.py`, `docker_manager.py`, `settings_manager.py`).
- `src/templates/`: Jinja2 templates (`index.html`, `settings.html`).
- `src/static/`: Frontend assets (`style.css`, `script.js`, `settings.js`, `favicon.ico`).
- `src/settings.json`: Runtime settings file.
- `src/uploads/`: Uploaded videos and generated outputs (served via `/videos/...`).
- `test.py`, `test.jsonl`: Root-level ad-hoc test script and sample data.
- `Dockerfile`: Production image (Python 3.13, ffmpeg, OpenCV, uvicorn entrypoint).

## Coding Style & Naming Conventions
- Python 3.13; follow PEP 8 with 4‑space indentation and type hints where practical.
- File encoding/line endings: use UTF-8 and CRLF.
- Comments must be written in Korean.
- Do not remove existing comments in code.
- Keep FastAPI endpoints small; move OCR logic into `src/core/`.
- Prefer explicit env access (e.g., `DOCKER_URL`, `DOCKER_NAME`, `KAFKA_URL`) with safe defaults.

