"""Text extraction from web articles and text/PDF files."""

from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
_USER_AGENT = "financial-sentiment-analyzer/0.1"


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "aside"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "aside"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not self._ignored_depth and text:
            self.parts.append(text)


def _download(url: str) -> tuple[bytes, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")

    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=20) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > _MAX_DOWNLOAD_BYTES:
            raise ValueError("Remote file is larger than 20 MB")
        data = response.read(_MAX_DOWNLOAD_BYTES + 1)
        if len(data) > _MAX_DOWNLOAD_BYTES:
            raise ValueError("Remote file is larger than 20 MB")
        return data, response.headers.get_content_type()


def _pdf_text(data: bytes) -> str:
    from PyPDF2 import PdfReader

    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
    if not text.strip():
        raise ValueError("The PDF contains no extractable text")
    return text


def read_article(url: str) -> str:
    """Return visible text extracted from an HTTP(S) article."""
    data, content_type = _download(url)
    if content_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError(f"Expected an HTML article, received {content_type}")

    parser = _ArticleParser()
    parser.feed(data.decode("utf-8", errors="replace"))
    text = "\n".join(parser.parts)
    if not text.strip():
        raise ValueError("The article contains no readable text")
    return text


def read_file(path_or_url: str) -> str:
    """Return text from a local or HTTP(S) TXT/PDF file."""
    parsed = urlparse(str(path_or_url))
    if parsed.scheme in {"http", "https"}:
        data, content_type = _download(str(path_or_url))
        suffix = Path(parsed.path).suffix.lower()
    else:
        path = Path(path_or_url).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.stat().st_size > _MAX_DOWNLOAD_BYTES:
            raise ValueError("File is larger than 20 MB")
        data, content_type, suffix = path.read_bytes(), "", path.suffix.lower()

    if suffix == ".pdf" or content_type == "application/pdf":
        return _pdf_text(data)
    if suffix in {".txt", ".md"} or content_type.startswith("text/plain"):
        text = data.decode("utf-8", errors="replace")
        if not text.strip():
            raise ValueError("The file is empty")
        return text
    raise ValueError("Only .txt, .md, and .pdf files are supported")
