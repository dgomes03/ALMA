import os
import pickle
import re
import unicodedata


# SDPA attention on the MPS backend (torch 2.12) caches Metal kernels/workspaces
# per padded sequence shape and never evicts them, so encoding a corpus with
# varying batch lengths leaks unbounded "other allocations" memory until OOM.
# Eager attention avoids that path entirely; throughput is equivalent at <=512 tokens.
_ST_MODEL_KWARGS = {"attn_implementation": "eager"}


# Emoji and pictographs, plus the joiner and variation selector that combine
# them. Check and cross marks (U+2713-2717) stay: answers use them in tables.
# The system prompt asks for no emojis; small models add them anyway.
_EMOJI = "[\U0001F000-\U0001FAFF\u2600-\u2712\u2718-\u27BF\uFE0F\u200D]+"
_EMOJI_BEFORE_PUNCT_RE = re.compile(f" ?{_EMOJI}(?=[.,!?;:]|\\s*$)")   # "feliz 😊!" -> "feliz!"
_EMOJI_RE = re.compile(f"{_EMOJI} ?")                                   # "😊 Olá"  -> "Olá"


def strip_emoji(text):
    """`text` without emojis. Only each emoji and the space beside it are removed."""
    return _EMOJI_RE.sub("", _EMOJI_BEFORE_PUNCT_RE.sub("", text))

def load_sentence_transformer(model_name):
    """Load a SentenceTransformer with MPS-safe attention (see _ST_MODEL_KWARGS)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, model_kwargs=_ST_MODEL_KWARGS)


def load_cross_encoder(model_name):
    """Load a CrossEncoder with MPS-safe attention (see _ST_MODEL_KWARGS)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, model_kwargs=_ST_MODEL_KWARGS)


def save_pickle(obj, filename):
    """Save object to pickle file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    """Load object from pickle file"""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None


# ---------------------------------------------------------------------------
# Text processing utilities for indexing and retrieval
# ---------------------------------------------------------------------------

# Lazy imports for NLTK/PyStemmer to avoid loading unless needed
_stemmer_cache = {}
_stop_words_cache = {}        # per-language stop word sets
_stop_words_union_cache = {}  # keyed by tuple(languages)

# Fallback if NLTK is not installed (English only)
_MINIMAL_ENGLISH_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'this', 'that', 'these', 'those', 'it', 'its'
}


def _get_stop_words_for_language(language):
    """Lazy load the stop word set for a single language."""
    if language not in _stop_words_cache:
        try:
            from nltk.corpus import stopwords
            import nltk
            try:
                _stop_words_cache[language] = set(stopwords.words(language))
            except LookupError:
                nltk.download('stopwords', quiet=True)
                _stop_words_cache[language] = set(stopwords.words(language))
        except ImportError:
            _stop_words_cache[language] = (
                _MINIMAL_ENGLISH_STOPWORDS if language == 'english' else set()
            )
        except OSError:
            # Language not available in NLTK
            _stop_words_cache[language] = set()
    return _stop_words_cache[language]


def _get_stop_words(languages):
    """Union of stop words across `languages`, cached per language combination."""
    key = tuple(languages)
    if key not in _stop_words_union_cache:
        union = set()
        for lang in languages:
            union |= _get_stop_words_for_language(lang)
        _stop_words_union_cache[key] = union
    return _stop_words_union_cache[key]


def _get_stemmer(language='english'):
    """Return a batch stemming callable (tokens -> stemmed tokens) for `language`.

    Prefers PyStemmer (C-backed Snowball, ~10x faster); falls back to NLTK's
    SnowballStemmer. Returns None if neither supports the language.
    """
    if language not in _stemmer_cache:
        stem_fn = None
        try:
            import Stemmer
            stem_fn = Stemmer.Stemmer(language).stemWords
        except (ImportError, KeyError):
            try:
                from nltk.stem import SnowballStemmer
                nltk_stemmer = SnowballStemmer(language)
                stem_fn = lambda tokens: [nltk_stemmer.stem(t) for t in tokens]
            except (ImportError, ValueError):
                stem_fn = None
        _stemmer_cache[language] = stem_fn
    return _stemmer_cache[language]


def _detect_language(tokens, languages):
    """Pick the language whose stop words appear most often in `tokens`.

    Cheap heuristic for choosing the right stemmer per text. Returns None
    when no stop words match (e.g. short keyword-only queries), signalling
    that the language is ambiguous.
    """
    if len(languages) == 1:
        return languages[0]
    best_lang, best_hits = None, 0
    for lang in languages:
        stop_words = _get_stop_words_for_language(lang)
        hits = sum(1 for t in tokens if t in stop_words)
        if hits > best_hits:
            best_lang, best_hits = lang, hits
    return best_lang


def _stem_multilingual(tokens, languages):
    """Stem each token with every language's stemmer, keeping distinct variants.

    Used when language detection is ambiguous: the union of stems matches
    documents indexed under any of the configured languages.
    """
    stemmers = [s for s in (_get_stemmer(lang) for lang in languages) if s]
    if not stemmers:
        return tokens
    per_lang = [stem(tokens) for stem in stemmers]
    result = []
    for variants in zip(*per_lang):
        result.extend(dict.fromkeys(variants))  # dedupe per token, keep order
    return result


def tokenize_for_bm25(text, enable_stemming=True, enable_stopwords=True,
                      languages=None):
    """
    Tokenize text for BM25 indexing/retrieval with optional stemming and stop word removal.

    The text's language is detected via stop word frequency and the matching
    stemmer is applied, so Portuguese text is stemmed with Portuguese rules
    and English text with English rules. When detection is ambiguous (no stop
    words present, e.g. keyword-only queries), tokens are stemmed with every
    configured language and the distinct variants are kept, so the query
    matches documents indexed in any language.

    Args:
        text: Input text to tokenize
        enable_stemming: Whether to apply stemming (default True)
        enable_stopwords: Whether to remove stop words (default True)
        languages: List of languages for stop words and stemming
                   (default ['english', 'portuguese'])

    Returns:
        List of processed tokens
    """
    if languages is None:
        languages = ['english', 'portuguese']

    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())

    # Split into tokens
    tokens = text.split()

    # Detect language before stop word removal — detection relies on stop words
    language = _detect_language(tokens, languages) if enable_stemming else None

    # Remove stop words if enabled
    if enable_stopwords:
        stop_words = _get_stop_words(languages)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # Apply stemming if enabled
    if enable_stemming:
        if language is None:
            tokens = _stem_multilingual(tokens, languages)
        else:
            stem = _get_stemmer(language)
            if stem:
                tokens = stem(tokens)

    return tokens


def split_sentences(text):
    """Split text into sentences using syntok's language-agnostic segmenter.

    Correctly handles abbreviations ("Dr.", "e.g.", "et al.", "Fig. 2",
    "p. ex."), decimals and ordinals that a naive regex split breaks on,
    for both English and Portuguese. Falls back to the regex split if
    syntok is unavailable.

    Args:
        text: Text to split

    Returns:
        List of sentence strings (stripped, no empties)
    """
    if not text:
        return []

    try:
        from syntok import segmenter
    except ImportError:
        return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    sentences = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            s = "".join(tok.spacing + tok.value for tok in sentence).strip()
            if s:
                sentences.append(s)
    return sentences


def clean_text(text, source_format='pdf'):
    """
    Clean and normalize text for better embedding quality.

    Handles:
    - Unicode normalization (all formats)
    - Whitespace cleanup (all formats)
    - PDF hyphenation fixes (pdf only)
    - Header/footer removal patterns (pdf only)
    - OCR garbage character removal (image_ocr only)
    - Residual HTML entity stripping (html only)

    Args:
        text: Raw text from document extraction
        source_format: Format of the source document ('pdf', 'html', 'image_ocr', or other)

    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""

    # Unicode normalization (NFKC for compatibility) — all formats
    text = unicodedata.normalize('NFKC', text)

    # Format-specific cleaning
    if source_format == 'pdf':
        # Fix PDF hyphenation (word-\n continuation)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # Fix line breaks within sentences
        text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)

        # Remove page numbers — ONLY at the very start or end of the block.
        #
        # Deleting EVERY standalone numeric line destroys data on tabular PDFs,
        # where codes and values often sit alone on their own line. A running
        # header/footer is only ever the first or last line of a page, so
        # anchoring there keeps the intent while leaving table cells alone.
        text = re.sub(r'\A(?:\s*\d{1,4}\s*\n)+', '', text)
        text = re.sub(r'(?:\n\s*\d{1,4}\s*)+\Z', '', text)

        # Remove common header/footer patterns
        text = re.sub(r'^\s*Page\s+\d+\s*(of\s+\d+)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    elif source_format == 'image_ocr':
        # Strip garbage characters common from OCR
        text = re.sub(r'[^\w\s.,;:!?\'\"()\-/&@#%$€£\n]', '', text)

    elif source_format == 'html':
        # Strip residual HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)

    # Normalize whitespace — all formats
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double

    return text.strip()


def prepare_for_embedding(text, is_query=False, use_prefix=True):
    """
    Prepare text for E5 embedding model with instruction prefixes.

    E5 models are trained with specific prefixes:
    - "query: " for search queries
    - "passage: " for documents/passages being indexed

    Args:
        text: Text to prepare
        is_query: True if this is a search query, False for passages
        use_prefix: Whether to add prefix (can be disabled via config)

    Returns:
        Text with appropriate prefix (or unchanged if use_prefix=False)
    """
    if not use_prefix:
        return text

    prefix = "query: " if is_query else "passage: "
    return prefix + text
