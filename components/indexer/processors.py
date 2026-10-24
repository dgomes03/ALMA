"""
Document processors for multi-format support.

Text extraction goes through unstructured-io's `partition(strategy="auto")` for the
common formats (PDF, DOCX, XLSX, PPTX, HTML, MD). Tables and embedded images are
extracted by a side-helper using format-native libraries — PyMuPDF for PDFs,
python-docx for DOCX, python-pptx for PPTX — and OCR'd via tesseract (eng+por).

Each processor function takes a file_info tuple and returns (doc_chunks, doc_metadata).

file_info = (
    file_path, doc_id, max_chunk_length, chunk_overlap,
    enable_ocr, ocr_min_width, ocr_min_height,
)

`enable_ocr` toggles standalone image files (.png/.jpg/...) and embedded-image OCR.
`ocr_min_width` / `ocr_min_height` filter embedded images below those pixel dimensions.
"""

import io
import os
import csv
import re

import pytesseract
from PIL import Image

import pymupdf as fitz  # the `fitz` import name is deprecated
from unstructured.partition.auto import partition

from core.utils import clean_text
from core.config import (
    PDF_CROSS_PAGE_CHUNKS, PDF_SECTION_PATH, PDF_SECTION_MAX_CHARS,
    PDF_SECTION_REQUIRE_NUMBER, PDF_TEXT_BACKEND,
    FRONT_MATTER_MAX_PAGE, FRONT_MATTER_MIN_TOKENS, FRONT_MATTER_UPPER_RATIO,
)
from components.indexer.standard import Indexer
from core.logging_config import get_logger

logger = get_logger(__name__)

# The Windows Tesseract installer does not add itself to PATH; use its default location.
if os.name == 'nt':
    import shutil
    _win_tesseract = os.path.join(os.environ.get('ProgramFiles', r'C:\Program Files'),
                                  'Tesseract-OCR', 'tesseract.exe')
    if not shutil.which('tesseract') and os.path.exists(_win_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _win_tesseract


_SECTION_NUM_RE = re.compile(r'^(\d+(?:\.\d+)*)(?=[\s.)\-]|$)')


def _heading_depth(title):
    """Nesting depth of a heading, inferred from a leading number.

    "5" -> 1, "5.3" -> 2, "5.3.1" -> 3. Unnumbered headings are treated as
    top level. `unstructured` gives no heading level for PDFs, so the numbering
    is the only structural signal available.
    """
    m = _SECTION_NUM_RE.match(title)
    return len(m.group(1).split('.')) if m else 1


def _is_heading_candidate(text):
    """Filter `unstructured`'s over-eager Title elements on PDFs.

    `unstructured` labels many wrapped body-text lines as Title on PDFs, and a
    length/punctuation guard alone lets most of them through. Requiring a
    leading section number ("5", "5.3", "5.3.1") is the signal that actually
    separates real headings, and it also tells apart sections whose titles
    are otherwise worded the same.

    Set PDF_SECTION_REQUIRE_NUMBER = False to also accept unnumbered headings,
    which raises recall of headings at a large cost in precision.
    """
    if not text or len(text) > PDF_SECTION_MAX_CHARS:
        return False
    if text.endswith('.'):
        return False
    m = _SECTION_NUM_RE.match(text)
    if PDF_SECTION_REQUIRE_NUMBER:
        # Needs a DOTTED number and a title after it: "5.3 Metodo ..." qualifies,
        # "5." and "1985" do not, and neither do chart axis tick labels like
        # "20 hdom" / "15 Idade (anos)", which a bare-integer rule accepted and
        # which then poisoned the root of every subsequent section path.
        return bool(m) and '.' in m.group(1) and len(text[m.end():].strip()) >= 3
    if m:
        return len(text[m.end():].strip()) >= 3
    # Unnumbered: must read like a heading, not a wrapped sentence fragment
    return text[:1].isupper() and not text.endswith((',', ';', ':')) and len(text) <= 80


def _push_heading(stack, title):
    """Update the heading stack in place and return the ancestor path."""
    depth = _heading_depth(title)
    while stack and stack[-1][0] >= depth:
        stack.pop()
    stack.append((depth, title))
    return ' > '.join(t for _, t in stack)


# Below this many characters per page a PDF is assumed to be scanned images,
# and extraction falls back to unstructured (which runs OCR via strategy="auto").
_MIN_CHARS_PER_PAGE = 100


def _is_front_matter(page_idx, page_text):
    """Is this early PDF page cover/title/TOC material rather than body text?

    Two signals, either sufficient: the page is short, or it is mostly
    upper-case (title pages, ministry letterheads, section dividers).
    Used only to DOWN-WEIGHT chunks at retrieval time, never to drop them.
    """
    if page_idx > FRONT_MATTER_MAX_PAGE:
        return False
    tokens = page_text.split()
    if len(tokens) < FRONT_MATTER_MIN_TOKENS:
        return True
    alpha = [t for t in tokens if any(ch.isalpha() for ch in t)]
    if not alpha:
        return True
    upper = sum(1 for t in alpha if t.isupper())
    return upper / len(alpha) > FRONT_MATTER_UPPER_RATIO


def _pdf_text_pieces(file_path):
    """Extract PDF text with PyMuPDF, grouped into (page, section_path) runs.

    Returns a list of ((page_index, section_path), [lines]) in document order,
    or None if the PDF looks scanned (so the caller falls back to unstructured,
    which will OCR it).

    Preferred over unstructured for PDFs because it preserves reading order —
    see PDF_TEXT_BACKEND in core/config.py for the measured failure it fixes.
    """
    pieces = []
    stack = []
    section = ''
    total_chars = 0
    doc = fitz.open(file_path)
    try:
        n_pages = doc.page_count
        for page_idx, page in enumerate(doc):
            text = page.get_text('text')
            total_chars += len(text)
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if PDF_SECTION_PATH and _is_heading_candidate(line):
                    section = _push_heading(stack, line)
                key = (page_idx, section)
                if pieces and pieces[-1][0] == key:
                    pieces[-1][1].append(line)
                else:
                    pieces.append((key, [line]))
    finally:
        doc.close()

    if n_pages and total_chars / n_pages < _MIN_CHARS_PER_PAGE:
        return None  # likely scanned; let unstructured OCR it
    return pieces


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_metadata(filename, doc_id, **kwargs):
    """Build a consistent metadata dict for any chunk."""
    meta = {
        'filename': filename,
        'doc_id': doc_id,
        'document_name': filename,
        'full_document_id': f"{filename}_{doc_id}",
    }
    meta.update(kwargs)
    return meta


def _fmt_label(ext):
    """Map file extension to a `clean_text` source_format label."""
    if ext == '.pdf':
        return 'pdf'
    if ext in ('.html', '.htm'):
        return 'html'
    if ext == '.md':
        return 'markdown'
    if ext == '.docx':
        return 'docx'
    if ext == '.pptx':
        return 'pptx'
    if ext in ('.xlsx', '.xls'):
        return 'xlsx'
    return 'text'


def _scope_for(el, ext, sheet_index_map, current_section):
    """Determine (scope_key, scope_meta) for a partitioned element.

    The scope key groups text-bearing elements into the same buffer so that
    chunking happens *within* a page (PDF / PPTX) / sheet (XLSX) / section (MD)
    / whole-doc (DOCX / HTML), matching the previous per-format behavior.
    """
    md = getattr(el, 'metadata', None)
    page_number = getattr(md, 'page_number', None) if md else None

    if ext == '.pdf':
        idx = (page_number - 1) if page_number else 0
        return idx, {'page': idx}

    if ext == '.pptx':
        slide_idx = (page_number - 1) if page_number else 0
        return slide_idx, {
            'page': slide_idx,
            'slide': slide_idx + 1,
            'slide_index': slide_idx,
        }

    if ext in ('.xlsx', '.xls'):
        sheet_name = getattr(md, 'page_name', None) or ''
        if sheet_name not in sheet_index_map:
            sheet_index_map[sheet_name] = len(sheet_index_map)
        return sheet_name, {
            'page': 0,
            'sheet': sheet_name,
            'sheet_index': sheet_index_map[sheet_name],
        }

    if ext == '.md':
        return current_section, {'page': 0, 'section': current_section}

    # .docx, .html, .htm — single scope per doc
    return 0, {'page': 0}


# ---------------------------------------------------------------------------
# Side-helper: tables + embedded images (PDF / DOCX / PPTX)
# ---------------------------------------------------------------------------

def _ocr_pil(pil_img, min_w, min_h):
    """OCR a PIL image with eng+por tesseract. Returns cleaned text or ''."""
    if pil_img.width < min_w or pil_img.height < min_h:
        return ''
    try:
        text = pytesseract.image_to_string(pil_img, lang='eng+por').strip()
    except Exception:
        return ''
    return clean_text(text, source_format='image_ocr')


def _extract_tables_and_images(file_info):
    """Dispatch tables + image OCR for PDF/DOCX/PPTX via format-native libs."""
    file_path = file_info[0]
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return _scrape_pdf(file_info)
    if ext == '.docx':
        return _scrape_docx(file_info)
    if ext == '.pptx':
        return _scrape_pptx(file_info)
    return [], []


def _scrape_pdf(file_info):
    file_path, doc_id, _maxlen, _overlap, enable_ocr, ocr_min_w, ocr_min_h = file_info
    filename = os.path.basename(file_path)
    chunks, metadata = [], []
    doc = fitz.open(file_path)
    try:
        for page_idx, page in enumerate(doc):
            # Tables
            try:
                tabs = page.find_tables()
            except Exception:
                tabs = []
            tab_iter = getattr(tabs, 'tables', tabs)
            for table_idx, tbl in enumerate(tab_iter):
                try:
                    rows = tbl.extract()
                except Exception:
                    continue
                lines = [' | '.join((c or '').strip() for c in row) for row in rows if row]
                tbl_text = '\n'.join(lines).strip()
                if tbl_text:
                    chunks.append(tbl_text)
                    metadata.append(_build_metadata(
                        filename, doc_id, type='table',
                        page=page_idx, table_idx=table_idx))

            # Embedded images → OCR
            if not enable_ocr:
                continue
            for img_idx, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.alpha or pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    pil_img = Image.open(io.BytesIO(pix.tobytes('png')))
                    ocr = _ocr_pil(pil_img, ocr_min_w, ocr_min_h)
                    if ocr:
                        chunks.append(ocr)
                        metadata.append(_build_metadata(
                            filename, doc_id, type='image_ocr',
                            page=page_idx, img_idx=img_idx))
                except Exception:
                    pass
    finally:
        doc.close()
    return chunks, metadata


def _scrape_docx(file_info):
    file_path, doc_id, _maxlen, _overlap, enable_ocr, ocr_min_w, ocr_min_h = file_info
    filename = os.path.basename(file_path)
    chunks, metadata = [], []
    from docx import Document
    doc = Document(file_path)

    for table_idx, table in enumerate(doc.tables):
        rows = []
        for row in table.rows:
            rows.append(' | '.join(cell.text.strip() for cell in row.cells))
        tbl_text = '\n'.join(rows).strip()
        if tbl_text:
            chunks.append(tbl_text)
            metadata.append(_build_metadata(
                filename, doc_id, type='table',
                page=0, table_idx=table_idx))

    if enable_ocr:
        img_idx = 0
        for rel in doc.part.rels.values():
            if 'image' not in rel.reltype:
                continue
            try:
                pil_img = Image.open(io.BytesIO(rel.target_part.blob))
            except Exception:
                continue
            ocr = _ocr_pil(pil_img, ocr_min_w, ocr_min_h)
            if ocr:
                chunks.append(ocr)
                metadata.append(_build_metadata(
                    filename, doc_id, type='image_ocr',
                    page=0, img_idx=img_idx))
                img_idx += 1
    return chunks, metadata


def _scrape_pptx(file_info):
    file_path, doc_id, _maxlen, _overlap, enable_ocr, ocr_min_w, ocr_min_h = file_info
    filename = os.path.basename(file_path)
    chunks, metadata = [], []
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(file_path)
    for slide_idx, slide in enumerate(prs.slides):
        slide_table_idx = 0
        slide_img_idx = 0
        for shape in slide.shapes:
            if shape.has_table:
                rows = []
                for row in shape.table.rows:
                    rows.append(' | '.join(cell.text.strip() for cell in row.cells))
                tbl_text = '\n'.join(rows).strip()
                if tbl_text:
                    chunks.append(tbl_text)
                    metadata.append(_build_metadata(
                        filename, doc_id, type='table',
                        page=slide_idx, slide=slide_idx + 1,
                        slide_index=slide_idx, table_idx=slide_table_idx))
                    slide_table_idx += 1
            elif enable_ocr and shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    pil_img = Image.open(io.BytesIO(shape.image.blob))
                except Exception:
                    continue
                ocr = _ocr_pil(pil_img, ocr_min_w, ocr_min_h)
                if ocr:
                    chunks.append(ocr)
                    metadata.append(_build_metadata(
                        filename, doc_id, type='image_ocr',
                        page=slide_idx, slide=slide_idx + 1,
                        slide_index=slide_idx, img_idx=slide_img_idx))
                    slide_img_idx += 1
    return chunks, metadata


# ---------------------------------------------------------------------------
# Core: Unstructured-based processor
#   handles .pdf .docx .xlsx .xls .pptx .html .htm .md
# ---------------------------------------------------------------------------

def _process_with_unstructured(file_info):
    file_path, doc_id, max_chunk_length, chunk_overlap, enable_ocr, *_ocr_thresholds = file_info
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    chunks = []
    metadata = []

    try:
        # For PDFs, PyMuPDF preserves reading order where unstructured
        # does not. Falls back automatically when the PDF looks scanned.
        prebuilt_pdf_pieces = None
        if ext == '.pdf' and PDF_TEXT_BACKEND == 'pymupdf':
            prebuilt_pdf_pieces = _pdf_text_pieces(file_path)
            if prebuilt_pdf_pieces is None:
                logger.info(f"  {filename}: little extractable text, "
                      f"falling back to unstructured (OCR)")

        elements = [] if prebuilt_pdf_pieces is not None else partition(
            filename=file_path, strategy="auto", languages=["eng", "por"])

        text_by_scope = {}    # scope_key -> list[str] of element texts
        scope_order = []      # list of (scope_key, scope_meta) in first-seen order
        table_counter = {}    # scope_key -> running table index
        image_counter = {}    # scope_key -> running image index
        sheet_index_map = {}  # XLSX only: sheet_name -> first-seen index
        current_section = ''  # MD only: most-recent section heading

        # PDF structural chunking: buffer (page, section_path)-keyed runs in
        # document order so chunks can span page breaks while keeping the page
        # they start on. Cleaning still happens per run, as it did per page.
        is_pdf = ext == '.pdf'
        # The PyMuPDF path always routes through pdf_pieces; the unstructured
        # path only does so when cross-page chunking is on.
        pdf_paged = is_pdf and (PDF_CROSS_PAGE_CHUNKS
                                or prebuilt_pdf_pieces is not None)
        pdf_pieces = prebuilt_pdf_pieces or []
        heading_stack = []    # list of (depth, title)
        pdf_section = ''      # current ancestor path

        is_xlsx = ext in ('.xlsx', '.xls')
        # PDF/DOCX/PPTX tables come from the side-helper (format-native libs);
        # ignore Unstructured's Table elements there to avoid duplication.
        skip_unstructured_tables = ext in ('.pdf', '.docx', '.pptx')

        for el in elements:
            category = getattr(el, 'category', '') or el.__class__.__name__

            # PDF heading tracking. Unlike the .md branch below this does NOT
            # consume the element: a PDF heading is also body text, and dropping
            # it would orphan the prose that follows.
            if is_pdf and PDF_SECTION_PATH and category == 'Title':
                title = str(el).strip()
                if _is_heading_candidate(title):
                    pdf_section = _push_heading(heading_stack, title)

            # Markdown section tracking: Title elements drive `section` metadata.
            if ext == '.md' and category == 'Title':
                section = str(el).strip()
                if section:
                    current_section = section
                continue

            scope_key, scope_meta = _scope_for(el, ext, sheet_index_map, current_section)

            if category == 'Table':
                if skip_unstructured_tables:
                    continue
                tbl_text = str(el).strip()
                if not tbl_text:
                    continue
                if is_xlsx:
                    # XLSX: tables ARE the sheet content — route through the
                    # text buffer so sub-chunking + type='table' happens at flush.
                    if scope_key not in text_by_scope:
                        text_by_scope[scope_key] = []
                        scope_order.append((scope_key, scope_meta))
                    text_by_scope[scope_key].append(tbl_text)
                else:
                    # PDF / DOCX / PPTX / HTML: one chunk per table.
                    table_idx = table_counter.get(scope_key, 0)
                    table_counter[scope_key] = table_idx + 1
                    chunks.append(tbl_text)
                    metadata.append(_build_metadata(
                        filename, doc_id, type='table',
                        table_idx=table_idx, **scope_meta))

            elif category == 'Image':
                # PDF/DOCX/PPTX images come from the side-helper; ignore here.
                if skip_unstructured_tables:
                    continue
                if not enable_ocr:
                    continue
                ocr_text = str(el).strip()
                if not ocr_text:
                    continue
                cleaned = clean_text(ocr_text, source_format='image_ocr')
                if not cleaned:
                    continue
                img_idx = image_counter.get(scope_key, 0)
                image_counter[scope_key] = img_idx + 1
                chunks.append(cleaned)
                metadata.append(_build_metadata(
                    filename, doc_id, type='image_ocr',
                    img_idx=img_idx, **scope_meta))

            else:
                # All other text-bearing elements (NarrativeText, Title,
                # ListItem, UncategorizedText, Header, Footer, etc.)
                text = str(el).strip()
                if not text:
                    continue
                if pdf_paged:
                    key = (scope_meta['page'], pdf_section)
                    if pdf_pieces and pdf_pieces[-1][0] == key:
                        pdf_pieces[-1][1].append(text)
                    else:
                        pdf_pieces.append((key, [text]))
                    continue
                if scope_key not in text_by_scope:
                    text_by_scope[scope_key] = []
                    scope_order.append((scope_key, scope_meta))
                text_by_scope[scope_key].append(text)

        # Flush text buffers: aggregate per scope, clean, sub-chunk.
        fmt_label = _fmt_label(ext)
        chunk_type = 'table' if is_xlsx else 'text'

        if pdf_paged and pdf_pieces:
            # Clean each (page, section) run exactly as the per-page path did,
            # then chunk the whole document as one stream. The key rides through
            # the chunker untouched, so each chunk reports where it STARTS.
            pieces = []
            for key, texts in pdf_pieces:
                cleaned = clean_text('\n'.join(texts), source_format=fmt_label)
                if cleaned:
                    pieces.append((cleaned, key))

            # Front-matter detection needs whole-page text, so reassemble it
            # from the per-(page, section) runs before chunking splits it up.
            page_text = {}
            for text, (page, _section) in pieces:
                page_text[page] = page_text.get(page, '') + ' ' + text
            front_pages = {pg for pg, txt in page_text.items()
                           if _is_front_matter(pg, txt)}

            if PDF_CROSS_PAGE_CHUNKS:
                streams = [pieces]          # whole document as one stream
            else:
                streams = []                # legacy: never cross a page break
                for piece in pieces:
                    if streams and streams[-1][0][1][0] == piece[1][0]:
                        streams[-1].append(piece)
                    else:
                        streams.append([piece])

            chunk_idx = 0
            for stream in streams:
                for chunk, (page, section_path) in \
                        Indexer.chunk_text_with_overlap_paged(
                            stream, max_chunk_length, chunk_overlap):
                    if not chunk:
                        continue
                    extra = {}
                    if page in front_pages:
                        extra['is_front_matter'] = True
                    if section_path:
                        extra['section_path'] = section_path
                        # Must be in the EMBEDDED text, not only metadata, or it
                        # cannot influence retrieval. Deliberate exception to the
                        # "no filename in the embedding" rule: a section path is
                        # disambiguating content, a filename is not.
                        chunk = f"{section_path}\n\n{chunk}"
                    chunks.append(chunk)
                    metadata.append(_build_metadata(
                        filename, doc_id, type='text', page=page,
                        chunk_idx=chunk_idx, **extra))
                    chunk_idx += 1

        for scope_key, scope_meta in scope_order:
            raw = '\n'.join(text_by_scope[scope_key])
            cleaned = clean_text(raw, source_format=fmt_label)
            if not cleaned:
                continue
            for chunk_idx, chunk in enumerate(
                    Indexer.chunk_text_with_overlap(cleaned, max_chunk_length, chunk_overlap)):
                if chunk:
                    chunks.append(chunk)
                    metadata.append(_build_metadata(
                        filename, doc_id, type=chunk_type,
                        chunk_idx=chunk_idx, **scope_meta))

    except Exception as e:
        print(f"Could not read {filename}: {e}")

    # Side-helper: tables + embedded images for PDF/DOCX/PPTX (Unstructured's
    # auto strategy skips both for text-PDFs and produces lossy results for the
    # office formats; format-native libs do better and faster).
    try:
        side_chunks, side_meta = _extract_tables_and_images(file_info)
        chunks.extend(side_chunks)
        metadata.extend(side_meta)
    except Exception as e:
        logger.warning(f"  side-helper failed on {filename}: {e}")

    logger.info(f"Processed {filename}")
    return chunks, metadata


# ---------------------------------------------------------------------------
# Native: CSV
# ---------------------------------------------------------------------------

def process_csv(file_info):
    file_path, doc_id, max_chunk_length, chunk_overlap, *_ = file_info
    chunks = []
    metadata = []
    filename = os.path.basename(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            rows = [' | '.join(row) for row in csv.reader(f)]
        table_text = '\n'.join(rows).strip()
        if table_text:
            for chunk_idx, chunk in enumerate(
                    Indexer.chunk_text_with_overlap(table_text, max_chunk_length, chunk_overlap)):
                if chunk:
                    chunks.append(chunk)
                    metadata.append(_build_metadata(
                        filename, doc_id, type='table',
                        page=0, chunk_idx=chunk_idx))
        logger.info(f"Processed {filename}")
    except Exception as e:
        print(f"Could not read {filename}: {e}")
    return chunks, metadata


# ---------------------------------------------------------------------------
# Native: Plain text
# ---------------------------------------------------------------------------

def process_plaintext(file_info):
    file_path, doc_id, max_chunk_length, chunk_overlap, *_ = file_info
    chunks = []
    metadata = []
    filename = os.path.basename(file_path)
    try:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        text = clean_text(text, source_format='text')
        if text:
            for chunk_idx, chunk in enumerate(
                    Indexer.chunk_text_with_overlap(text, max_chunk_length, chunk_overlap)):
                if chunk:
                    chunks.append(chunk)
                    metadata.append(_build_metadata(
                        filename, doc_id, type='text',
                        page=0, chunk_idx=chunk_idx))
        logger.info(f"Processed {filename}")
    except Exception as e:
        print(f"Could not read {filename}: {e}")
    return chunks, metadata


# ---------------------------------------------------------------------------
# Native: Standalone images (tesseract OCR)
# ---------------------------------------------------------------------------

def process_image(file_info):
    file_path, doc_id, max_chunk_length, chunk_overlap, enable_ocr, *_ = file_info
    if not enable_ocr:
        return [], []
    chunks = []
    metadata = []
    filename = os.path.basename(file_path)
    try:
        pil_img = Image.open(file_path)
        ocr_text = pytesseract.image_to_string(pil_img, lang='eng+por').strip()
        ocr_text = clean_text(ocr_text, source_format='image_ocr')
        if ocr_text:
            produced = Indexer.chunk_text_with_overlap(ocr_text, max_chunk_length, chunk_overlap)
            if not produced:
                produced = [ocr_text]
            for chunk_idx, chunk in enumerate(produced):
                if chunk:
                    chunks.append(chunk)
                    metadata.append(_build_metadata(
                        filename, doc_id, type='image_ocr',
                        page=0, img_idx=chunk_idx))
        logger.info(f"Processed {filename}")
    except Exception as e:
        print(f"Could not read {filename}: {e}")
    return chunks, metadata


# ---------------------------------------------------------------------------
# Extension → processor mapping
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    '.pdf':  _process_with_unstructured,
    '.docx': _process_with_unstructured,
    '.xlsx': _process_with_unstructured,
    '.xls':  _process_with_unstructured,
    '.pptx': _process_with_unstructured,
    '.html': _process_with_unstructured,
    '.htm':  _process_with_unstructured,
    '.md':   _process_with_unstructured,
    '.csv':  process_csv,
    '.txt':  process_plaintext,
    '.png':  process_image,
    '.jpg':  process_image,
    '.jpeg': process_image,
    '.tiff': process_image,
    '.bmp':  process_image,
}


def process_single_document(file_info):
    """Top-level dispatch — picklable for `multiprocessing.Pool.imap`."""
    file_path = file_info[0]
    ext = os.path.splitext(file_path)[1].lower()
    processor = SUPPORTED_EXTENSIONS.get(ext)
    if processor is None:
        logger.info(f"Unsupported file format: {os.path.basename(file_path)} ({ext})")
        return [], []
    chunks, metadata = processor(file_info)
    # Defensive: HF tokenizer (Rust) rejects strings with NUL bytes or invalid
    # UTF-8 codepoints (lone surrogates). PyMuPDF table extraction occasionally
    # produces both. Round-trip through UTF-8 with errors='ignore' to clean.
    out_chunks, out_meta = [], []
    for c, m in zip(chunks, metadata):
        if not isinstance(c, str):
            try:
                c = str(c) if c is not None else ''
            except Exception:
                c = ''
        c = c.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8').strip()
        if c:
            out_chunks.append(c)
            out_meta.append(m)
    return out_chunks, out_meta
