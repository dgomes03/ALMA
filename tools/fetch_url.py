import re
import urllib.parse
import urllib.request

from core.config import MAX_FETCH_URL_CHARS


def _extract_text(html):
    """Extract the page's main content, dropping nav bars, cookie banners,
    footers and other boilerplate.

    Uses trafilatura; falls back to regex tag-stripping if it is unavailable
    or finds no main content (e.g. sparse or non-article pages).
    """
    try:
        import trafilatura
        text = trafilatura.extract(html, include_comments=False)
        if text:
            return text
    except ImportError:
        pass

    if isinstance(html, bytes):
        html = html.decode('utf-8', errors='ignore')
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.I)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.I)
    html = re.sub(r'<(nav|header|footer)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.I)
    text = re.sub(r'<[^>]+>', ' ', html)
    return re.sub(r'\s+', ' ', text).strip()


def fetch_url_content(url: str, max_chars: int = None) -> dict:
    """Fetch and extract main text content from a URL."""
    if max_chars is None:
        max_chars = MAX_FETCH_URL_CHARS
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ['http', 'https']:
            return {'error': 'Invalid URL scheme', 'url': url}

        headers = {'User-Agent': 'Mozilla/5.0 (compatible; RAGBot/1.0)'}
        request = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(request, timeout=15) as response:
            # Raw bytes: trafilatura does its own encoding detection
            html = response.read()

        text = _extract_text(html)

        if len(text) > max_chars:
            text = text[:max_chars] + '...'

        return {'url': url, 'content': text, 'char_count': len(text), 'success': True}
    except Exception as e:
        return {'error': str(e), 'url': url}
