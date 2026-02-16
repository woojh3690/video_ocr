import re
import unicodedata

BRACKET_CHARS = "[](){}<>「」『』〈〉《》"
BRACKET_PAIRS = {
    BRACKET_CHARS[i]: BRACKET_CHARS[i + 1]
    for i in range(0, len(BRACKET_CHARS), 2)
}

_WS_RE = re.compile(r"\s+")

# 공백·개행·탭 등을 하나의 공백으로 줄여 주는 헬퍼
def normalize_text(text: str) -> str:
    """Strip and collapse whitespace."""
    if not text:
        return ""
    text = text.strip()
    return _WS_RE.sub(" ", text)

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    
    # 노멀라이즈
    normalized = unicodedata.normalize("NFKC", text)

    # 양 끝에 있는 브라켓 제거
    while len(normalized) > 1:
        opening = normalized[0]
        closing = BRACKET_PAIRS.get(opening)
        if not closing or normalized[-1] != closing:
            break
        normalized = normalized[1:-1]
    return normalize_text(normalized)
