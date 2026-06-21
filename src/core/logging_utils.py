from datetime import datetime


def log_ocr_event(component: str, message: str, task_id: str | None = None) -> None:
    # OCR 로그를 시간, 구성요소, 작업 ID가 포함된 한 줄 형식으로 즉시 출력합니다.
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    task_context = f"[task={task_id}]" if task_id else ""
    print(f"{timestamp} [OCR][{component}]{task_context} {message}", flush=True)
