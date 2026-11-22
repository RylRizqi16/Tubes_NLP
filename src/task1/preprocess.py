"""Inference text cleaning aligned with build_master_dataset.clean_text."""
from __future__ import annotations

import html
import re

HTML_TAG_RE = re.compile(
    r"</?(?:p|br|div|span|a|b|i|u|strong|em|img|ul|ol|li|h[1-6]|table|tr|td|th|"
    r"thead|tbody|script|style|iframe|blockquote|pre|code|hr|font|center|small|"
    r"sub|sup|mark|section|article|header|footer|nav)\b[^>]*>", re.IGNORECASE,
)
URL_RE = re.compile(r"(?:https?://\S+|\bwww\.\S+)", re.IGNORECASE)
URL_PLACEHOLDER_RE = re.compile(r"<(?:link|url)>", re.IGNORECASE)
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\xad]")
CTRL_RE = re.compile(r"[\x00-\x1f\x7f]+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    result = html.unescape(text)
    result = ZERO_WIDTH_RE.sub("", result)
    result = HTML_TAG_RE.sub(" ", result)
    result = URL_PLACEHOLDER_RE.sub(" ", result)
    result = URL_RE.sub(" ", result)
    result = CTRL_RE.sub(" ", result)
    return WS_RE.sub(" ", result).strip()
