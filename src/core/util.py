import re
import unicodedata

_WS_RE = re.compile(r"\s+")

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    
    # 노멀라이즈
    normalized = unicodedata.normalize("NFKC", text)

    # 공백·개행·탭 등을 하나의 공백으로 줄여 주는 헬퍼
    """Strip and collapse whitespace."""
    if not normalized:
        return ""
    normalized = normalized.strip()
    return _WS_RE.sub(" ", normalized)
