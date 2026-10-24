from core.config import DUCKDUCKGO_RESULTS_COUNT


def duckduckgo_search(query: str, max_results: int = None) -> dict:
    """Search DuckDuckGo (free, no API key required)."""
    if max_results is None:
        max_results = DUCKDUCKGO_RESULTS_COUNT
    else:
        # Never return fewer results than the configured minimum
        max_results = max(DUCKDUCKGO_RESULTS_COUNT, max_results)
    try:
        DDGS = None
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return {
                    'error': 'DuckDuckGo search requires the ddgs package. Install with: pip install ddgs',
                    'query': query,
                }

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', ''),
                })

        return {'query': query, 'results': results, 'total_results': len(results)}
    except Exception as e:
        return {'error': str(e), 'query': query}
