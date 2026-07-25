#!/usr/bin/env python3
"""
rag_engine.py — RAG-движок базы знаний по ВЭД для бота DiV_executive.

«Прагматичный» уровень поверх эталона урока zerocoder «PEr01. Что такое RAG»:

  Базовые улучшения (уже лучше урока):
    • FAISS IndexFlatIP + L2-нормализация → косинусная близость (вместо L2);
    • батчинг эмбеддингов (вместо по одному);
    • инкрементальная индексация по sha256 (вместо полного пере-эмбеддинга);
    • ретраи с экспоненциальным backoff;
    • поддержка PDF (pypdf) + txt/md;
    • ВЭД-системный промпт + порог релевантности (антигаллюцинация).

  Прагматичный уровень (урок этого не делает):
    • структурный чанкинг с детектом заголовков «Статья/Глава/Раздел/§»
      и метаданными {source, heading, file_hash} — цитата до статьи;
    • гибридный поиск: BM25 (ключевые слова) + dense FAISS (смысл) + RRF-фьюжн
      (Reciprocal Rank Fusion) — ловит и точные термины («справка», «ст. 19»), и смысл;
    • опциональный реранкер qwen3-rerank (DashScope text-rerank) поверх кандидатов,
      с graceful-fallback на RRF-порядок при ошибке/недоступности;
    • мини eval-сет на 16 вопросов (recall источника + попадание ключевых слов +
      корректность «не найдено») — превращает «кажется хорошим» в измерение.

Стек: DashScope (Alibaba) — эмбеддинги text-embedding-v3 + чат qwen-plus +
реранк qwen3-rerank, OpenAI-совместимый эндпоинт для embed/chat; реранк через
нативный сервис DashScope (не входит в OpenAI-compat API). Ключи из .env.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ────────────────────────── зависимости (мягкий импорт) ──────────────────────────
try:
    import numpy as np
    import faiss
    _FAISS_OK = True
except Exception as e:  # pragma: no cover
    logger.warning(f"FAISS/numpy недоступны: {e}")
    _FAISS_OK = False

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception as e:  # pragma: no cover
    logger.warning(f"openai SDK недоступен: {e}")
    _OPENAI_OK = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except Exception as e:  # pragma: no cover
    logger.warning(f"rank_bm25 недоступен: {e}")
    _BM25_OK = False

try:
    from pypdf import PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        PdfReader = None  # type: ignore

try:
    import requests
    _REQUESTS_OK = True
except Exception:
    _REQUESTS_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ────────────────────────── конфигурация ──────────────────────────
BASE_DIR = Path(__file__).resolve().parent / "rag_data"
DOCS_DIR = BASE_DIR / "docs"
INDEX_PATH = BASE_DIR / "faiss_index.bin"
META_PATH = BASE_DIR / "metadata.json"
MANIFEST_PATH = BASE_DIR / "manifest.json"
EVAL_PATH = BASE_DIR / "eval_report.json"

API_KEY = (os.getenv("API_KEY") or os.getenv("OPENROUTER_API_KEY") or "").strip()
BASE_URL = (os.getenv("BASE_URL") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").strip()
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-v3")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "qwen3-vl-flash")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", "10"))
SIM_THRESHOLD = float(os.getenv("RAG_SIM_THRESHOLD", "0.30"))   # косинус, гейт dense
BM25_TOPN = int(os.getenv("RAG_BM25_TOPN", "8"))                 # гейт bm25 (топ-N проходит даже при низком dense)
CANDIDATE_POOL = int(os.getenv("RAG_CANDIDATE_POOL", "20"))      # кандидатов из каждого ретривера до реранка
MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT", "6000"))
REQUEST_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "60"))
ANSWER_VERIFY = os.getenv("RAG_ANSWER_VERIFY", "1").strip() in ("1", "true", "yes", "on")
VERIFY_MIN_GROUNDED_SCORE = 0.6

# реранкер (нативный сервис DashScope; не OpenAI-compat)
RERANK_ON = os.getenv("RAG_RERANK_ON", "1").strip() in ("1", "true", "yes", "on")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "qwen3-rerank")
RERANK_URL = os.getenv("RAG_RERANK_URL", "https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks")
RERANK_MIN_SCORE = float(os.getenv("RAG_RERANK_MIN_SCORE", "0.05"))

RAG_AVAILABLE = bool(_FAISS_OK and _OPENAI_OK and API_KEY)

# ────────────────────────── системный промпт ВЭД ──────────────────────────
RAG_SYSTEM_PROMPT = (
    "Ты — юрист-консультант по внешнеэкономической деятельности (ВЭД): таможенное оформление, "
    "валютный контроль, договоры и документы для международных сделок.\n"
    "Отвечай СТРОГО на основе переданного ниже КОНТЕКСТА из нормативных документов. "
    "Если ответ в контексте есть — сформулируй его ясно, со ссылками на источник в виде "
    "[источник: <файл>, <заголовок статьи, если есть>].\n"
    "ФОРМАТИРОВАНИЕ ОТВЕТА:\n"
    "- Используй plain-text Telegram: маркированные списки с дефисом «-», нумерованные списки с цифрой и точкой.\n"
    "- НЕ используй markdown-разметку (#, ##, **, *, `, >, ###).\n"
    "- Заголовки разделов выделяй простым текстом с двоеточием, например: «Документы:» или «Сведения в декларации:».\n"
    "- Если нужно выделить термин — используй кавычки или прописные буквы, не звёздочки.\n"
    "- Используй списки для перечней документов/шагов.\n"
    "Если предоставлено несколько источников — используй их все и укажи, чем они дополняют друг друга. "
    "Если в контексте НЕТ сведений для ответа — обязательно ответь дословно: "
    "«По имеющейся базе ответа не нашёл. Уточните вопрос или переиндексируйте документы (/ingest).» "
    "Не придумывай, не используй внешние знания и не додумывай положения, которых нет в контексте.\n"
    "В конце ответа добавь короткую строку-дисклеймер: «Не является юридической консультацией.»"
)

RAG_PROMPT_TEMPLATE = (
    "КОНТЕКСТ (нормативные документы ВЭД):\n{context}\n\n"
    "{history}"
    "Вопрос пользователя: {query}\n\n"
    "Ответь plain-text для Telegram (без markdown #, ##, **, *, `, >). Используй только дефисы и цифры для списков."
)

# ────────────────────────── OpenAI-клиент (синглтон) ──────────────────────────
_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT)
    return _client


# ────────────────────────── ретраи ──────────────────────────
_RETRYABLE = ("timeout", "timed out", "429", "rate", "connection",
              "unavailable", "temporar", "reset", "eof")


def _with_retries(fn, *args, **kwargs):
    """Экспоненциальный backoff: 4 попытки, 1/2/4/8 c + jitter. На ретраимые ошибки."""
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            retryable = any(k in msg for k in _RETRYABLE)
            if attempt == 3 or not retryable:
                raise
            sleep = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"RAG retry {attempt + 1}/4 через {sleep:.1f}с: {e}")
            time.sleep(sleep)
    raise last_err  # type: ignore[misc]


# ────────────────────────── эмбеддинги ──────────────────────────
def _embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _get_client()

    def call():
        return client.embeddings.create(model=EMBED_MODEL, input=texts, encoding_format="float")

    resp = _with_retries(call)
    data = sorted(resp.data, key=lambda d: d.index)
    return [list(map(float, d.embedding)) for d in data]


def _embed_all(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    n = len(texts)
    for i in range(0, n, EMBED_BATCH):
        out.extend(_embed_batch(texts[i:i + EMBED_BATCH]))
        logger.info(f"RAG embed: {min(i + EMBED_BATCH, n)}/{n} чанков")
    return out


def _normalize(vectors: "np.ndarray") -> "np.ndarray":
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


# ────────────────────────── чтение и структурный чанкинг ──────────────────────────
def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf не установлен: pip install pypdf")
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception as e:
                logger.warning(f"PDF {path.name}: пропуск страницы ({e})")
        text = "\n".join(parts).strip()
        if not text:
            logger.warning(f"PDF {path.name}: текстовый слой пуст (скан?) — пропущен")
        return text
    return ""


# детект заголовков:
# - markdown: # / ## / ### в начале строки
# - юридические: Статья 19, Глава 3, Раздел II, § 5, Ст. 19, п. 3, пункт 3
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"#{1,3}\s+.*|"                             # markdown headings
    r"статья\s+[\dIVXLCDM]+(?:\s+.*)?|"        # Статья 19. Название
    r"глава\s+[\dIVXLCDM]+(?:\s+.*)?|"        # Глава 3. ...
    r"раздел\s+[IVXLCDM\d]+(?:\s+.*)?|"       # Раздел II ...
    r"ст\.?\s*[\dIVXLCDM]+(?:\s+.*)?|"        # Ст. 19 ...
    r"гл\.?\s*[\dIVXLCDM]+(?:\s+.*)?|"        # Гл. 3 ...
    r"§\s*\d+(?:\s+.*)?|"                      # § 5 ...
    r"пункт\s+\d+(?:\s+.*)?"                   # пункт 3 ...
    r")\b",
    re.IGNORECASE,
)


def _window_chunks(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    step = max(1, size - overlap)
    i, n = 0, len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            # стараемся обрезать по концу абзаца/предложения
            for sep in ("\n\n", "\n", ". "):
                pos = text.rfind(sep, i, end)
                if pos > i + size // 2:
                    end = pos + len(sep)
                    break
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        i = end - overlap if end - overlap > i else end
    return chunks


def _extract_article(heading: str) -> str:
    """Из заголовка вида 'Статья 19. Название' извлекает 'Статья 19'."""
    m = re.search(r"(статья\s+[\dIVXLCDM]+|ст\.?\s*[\dIVXLCDM]+|§\s*\d+|пункт\s+\d+|глава\s+[\dIVXLCDM]+)", heading, re.IGNORECASE)
    return m.group(1) if m else ""


def _chunk_text(text: str) -> list[dict]:
    """
    Структурный чанкинг: разбиваем по заголовкам.
    Короткие секции оставляем цельными, длинные — дробим окном.
    Возвращаем [{text, heading, article}].
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    lines = text.split("\n")

    sections: list[tuple[str, str]] = []   # (heading, body)
    cur_heading = ""
    cur_body: list[str] = []

    def flush():
        body = "\n".join(cur_body).strip()
        if body:
            sections.append((cur_heading, body))

    for line in lines:
        stripped = line.strip()
        if stripped and _HEADING_RE.match(stripped) and len(stripped) < 120:
            flush()
            cur_heading = stripped
            cur_body = []
        else:
            cur_body.append(line)
    flush()

    chunks: list[dict] = []
    for heading, body in sections:
        article = _extract_article(heading)
        if len(body) <= CHUNK_SIZE:
            text_with_h = f"{heading}\n{body}".strip() if heading else body
            chunks.append({"text": text_with_h, "heading": heading, "article": article})
        else:
            for sub in _window_chunks(body, CHUNK_SIZE, CHUNK_OVERLAP):
                text_with_h = f"{heading}\n{sub}".strip() if heading else sub
                chunks.append({"text": text_with_h, "heading": heading, "article": article})
    return chunks


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ────────────────────────── BM25 (лексический ретривер) ──────────────────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower(), re.UNICODE)


_bm25: Any = None  # BM25Okapi над текстами чанков (порядок == _metadata)


def _build_bm25() -> None:
    global _bm25
    if not _BM25_OK or not _metadata:
        _bm25 = None
        return
    try:
        corpus = [_tokenize(m["text"]) for m in _metadata]
        _bm25 = BM25Okapi(corpus)
    except Exception as e:
        logger.warning(f"RAG BM25 не построен: {e}")
        _bm25 = None


def _bm25_search(query: str, k: int) -> list[tuple[int, float]]:
    """Топ-k (idx, bm25_score) по убыванию score."""
    if _bm25 is None or not _metadata:
        return []
    scores = _bm25.get_scores(_tokenize(query))
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


# ────────────────────────── реранкер (qwen3-rerank) ──────────────────────────
def _rerank(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]] | None:
    """
    Возвращает [(idx_into_docs, relevance_score)] отсортированно, или None при ошибке
    (тогда вызывающая сторона падает на RRF-порядок).
    """
    if not (RERANK_ON and RERANK_MODEL and _REQUESTS_OK and docs):
        return None
    # OpenAI-compatible rerank endpoint (DashScope intl: compatible-api/v1/reranks)
    payload = {"model": RERANK_MODEL, "query": query,
               "documents": docs, "top_n": min(top_n, len(docs))}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    def call():
        r = requests.post(RERANK_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r

    try:
        resp = _with_retries(call)
        data = resp.json()
        results = data.get("results") or (data.get("output", {}) or {}).get("results") or []
        out = [(int(r["index"]), float(r.get("relevance_score", r.get("score", 0.0))))
               for r in results]
        out.sort(key=lambda x: -x[1])
        return out
    except Exception as e:
        logger.warning(f"RAG rerank недоступен/ошибка — фоллбэк на RRF: {e}")
        return None


# ────────────────────────── векторное хранилище (FAISS, косинус) ──────────────────────────
_index: Any = None
_metadata: list[dict] = []
_dim: int | None = None


def _load_store() -> None:
    global _index, _metadata, _dim
    if not _FAISS_OK:
        return
    if INDEX_PATH.exists() and META_PATH.exists():
        try:
            _index = faiss.read_index(str(INDEX_PATH))
            _metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
            _dim = _index.d
            _build_bm25()
        except Exception as e:
            logger.warning(f"RAG: индекс не загружен ({e}) — начнём заново")
            _index, _metadata, _dim = None, [], None


def _ensure_index(dim: int) -> None:
    global _index, _dim
    if _index is None or _dim != dim:
        _index = faiss.IndexFlatIP(dim)
        _dim = dim
        _metadata.clear()


def _save_store() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if _index is not None:
        faiss.write_index(_index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(_metadata, ensure_ascii=False), encoding="utf-8")


def _reconstruct_vector(row_idx: int) -> "np.ndarray":
    return _index.reconstruct(row_idx)


# ────────────────────────── публичный API: ingest ──────────────────────────
def ingest(verbose: bool = False) -> dict:
    """Инкрементальная индексация DOCS_DIR → {added, updated, removed, total_chunks, files, errors}."""
    global _index, _metadata, _dim
    if not RAG_AVAILABLE:
        raise RuntimeError("RAG недоступен (faiss/openai/API_KEY)")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    _load_store()

    exts = {".txt", ".md", ".pdf"}
    current: dict[str, Path] = {p.name: p for p in DOCS_DIR.iterdir()
                                if p.is_file() and p.suffix.lower() in exts}

    old_by_source: dict[str, list[int]] = {}
    old_file_hash: dict[str, str] = {}
    for idx, m in enumerate(_metadata):
        old_by_source.setdefault(m["source"], []).append(idx)
        old_file_hash[m["source"]] = m.get("file_hash", "")

    added = updated = removed = 0
    errors: list[str] = []
    keep_vecs: list["np.ndarray"] = []
    keep_meta: list[dict] = []
    new_chunks: list[dict] = []          # {text, heading}
    new_meta_specs: list[dict] = []

    for name in list(old_by_source.keys()):
        if name not in current:
            removed += 1

    for name, path in current.items():
        try:
            h = _file_hash(path)
            if name in old_by_source and old_file_hash.get(name) == h and _index is not None:
                for row_idx in old_by_source[name]:
                    keep_vecs.append(_reconstruct_vector(row_idx))
                    m = _metadata[row_idx]
                    keep_meta.append({
                        "id": f"{name}#{m['chunk_id']}", "source": name,
                        "chunk_id": m["chunk_id"], "text": m["text"],
                        "heading": m.get("heading", ""),
                        "article": m.get("article", ""),
                        "file_hash": h,
                    })
                continue
            text = _read_document(path)
            if not text:
                errors.append(f"{name}: пустой/не читается")
                continue
            chunks = _chunk_text(text)
            for cid, ch in enumerate(chunks):
                new_chunks.append(ch)
                new_meta_specs.append({
                    "source": name, "chunk_id": cid,
                    "file_hash": h,
                    "heading": ch.get("heading", ""),
                    "article": ch.get("article", ""),
                })
            if name in old_by_source:
                updated += 1
            else:
                added += 1
        except Exception as e:
            errors.append(f"{name}: {e}")
            logger.error(f"RAG ingest {name}: {e}", exc_info=True)

    if new_chunks:
        vecs = _embed_all([c["text"] for c in new_chunks])
        dim = len(vecs[0])
        if keep_vecs and int(keep_vecs[0].shape[0]) != dim:
            logger.warning("RAG: размерность изменилась — полная переиндексация")
            return _full_reindex(current, errors)
        if _index is None or _dim != dim:
            _ensure_index(dim)
        new_arr = _normalize(np.array(vecs, dtype="float32"))
        for spec, vec, ch in zip(new_meta_specs, new_arr, new_chunks):
            keep_vecs.append(vec)
            keep_meta.append({
                "id": f"{spec['source']}#{spec['chunk_id']}", "source": spec["source"],
                "chunk_id": spec["chunk_id"], "text": ch["text"],
                "heading": spec["heading"], "article": spec["article"],
                "file_hash": spec["file_hash"],
            })

    if keep_vecs:
        dim = int(keep_vecs[0].shape[0])
        _index = faiss.IndexFlatIP(dim)
        _dim = dim
        _metadata = keep_meta
        _index.add(np.array(keep_vecs, dtype="float32"))
    else:
        _index = faiss.IndexFlatIP(_dim or 1024)
        _dim = _dim or 1024
        _metadata = []
    _save_store()
    _save_manifest(current)
    _build_bm25()

    return {
        "added": added, "updated": updated, "removed": removed,
        "total_chunks": len(_metadata),
        "files": sorted({m["source"] for m in _metadata}), "errors": errors,
    }


def _full_reindex(current: dict[str, Path], errors: list[str]) -> dict:
    global _index, _metadata, _dim
    all_chunks: list[dict] = []
    specs: list[dict] = []
    added = 0
    for name, path in current.items():
        try:
            text = _read_document(path)
            if not text:
                errors.append(f"{name}: пустой")
                continue
            for cid, ch in enumerate(_chunk_text(text)):
                all_chunks.append(ch)
                specs.append({
                    "source": name, "chunk_id": cid,
                    "file_hash": _file_hash(path),
                    "heading": ch.get("heading", ""),
                    "article": ch.get("article", ""),
                })
            added += 1
        except Exception as e:
            errors.append(f"{name}: {e}")
    if all_chunks:
        vecs = _embed_all([c["text"] for c in all_chunks])
        dim = len(vecs[0])
        arr = _normalize(np.array(vecs, dtype="float32"))
        _index = faiss.IndexFlatIP(dim)
        _dim = dim
        _metadata = [
            {"id": f"{s['source']}#{s['chunk_id']}", "source": s["source"],
             "chunk_id": s["chunk_id"], "text": ch["text"],
             "heading": s["heading"], "article": s["article"],
             "file_hash": s["file_hash"]}
            for s, ch in zip(specs, all_chunks)
        ]
        _index.add(arr)
    else:
        _index = faiss.IndexFlatIP(1024)
        _dim = 1024
        _metadata = []
    _save_store()
    _save_manifest(current)
    _build_bm25()
    return {"added": added, "updated": 0, "removed": 0,
            "total_chunks": len(_metadata),
            "files": sorted({m["source"] for m in _metadata}), "errors": errors}


def _save_manifest(current: dict[str, Path]) -> None:
    manifest: dict[str, dict] = {}
    for m in _metadata:
        manifest.setdefault(m["source"], {"hash": m.get("file_hash", ""), "chunks": 0})
        manifest[m["source"]]["chunks"] += 1
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# ────────────────────────── публичный API: get_stats ──────────────────────────
def get_stats() -> dict:
    if not RAG_AVAILABLE:
        return {"available": False, "empty": True, "collection_count": 0, "files": [],
                "models": {"embed": EMBED_MODEL, "chat": CHAT_MODEL,
                           "rerank": RERANK_MODEL if RERANK_ON else None},
                "bm25": _BM25_OK, "rerank_on": RERANK_ON,
                "docs_dir": str(DOCS_DIR)}
    if _index is None:
        _load_store()
    count = int(_index.ntotal) if _index is not None else 0
    files = []
    if MANIFEST_PATH.exists():
        try:
            m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            files = [{"name": k, "chunks": v.get("chunks", 0)} for k, v in m.items()]
        except Exception:
            pass
    return {
        "available": True, "empty": count == 0, "collection_count": count, "files": files,
        "models": {"embed": EMBED_MODEL, "chat": CHAT_MODEL,
                   "rerank": RERANK_MODEL if RERANK_ON else None},
        "bm25": _BM25_OK and _bm25 is not None, "rerank_on": RERANK_ON,
        "docs_dir": str(DOCS_DIR),
    }


# ────────────────────────── публичный API: query (гибридный) ──────────────────────────
def _hybrid_retrieve(query: str, k: int) -> list[dict]:
    """
    Гибридный поиск: dense FAISS + BM25 → RRF-фьюжн → (опц.) реранкер.
    Гейт релевантности: dense_sim >= SIM_THRESHOLD ИЛИ bm25_rank < BM25_TOPN.
    Возвращает список {idx, sim, bm25_rank, source, heading, text, rrf} отсорченно,
    уже после гейта; пустой = «не найдено».
    """
    pool = CANDIDATE_POOL

    # dense
    q_vec = _embed_batch([query])[0]
    q_arr = _normalize(np.array([q_vec], dtype="float32"))
    sims, ids = _index.search(q_arr, pool)
    dense_rank: dict[int, int] = {int(idx): r for r, idx in enumerate(ids[0]) if idx >= 0}
    dense_sim: dict[int, float] = {int(idx): float(s)
                                   for s, idx in zip(sims[0], ids[0]) if idx >= 0}

    # bm25
    bm25_hits = _bm25_search(query, pool)
    bm25_rank: dict[int, int] = {idx: r for r, (idx, _sc) in enumerate(bm25_hits)}

    # RRF
    rrf_k = 60
    candidates: set[int] = set(dense_rank) | set(bm25_rank)
    rrf_scores: list[tuple[float, int]] = []
    for idx in candidates:
        score = 0.0
        if idx in dense_rank:
            score += 1.0 / (rrf_k + dense_rank[idx] + 1)
        if idx in bm25_rank:
            score += 1.0 / (rrf_k + bm25_rank[idx] + 1)
        rrf_scores.append((score, idx))
    rrf_scores.sort(key=lambda x: -x[0])

    # гейт релевантности
    gated = []
    for rrf, idx in rrf_scores:
        sim = dense_sim.get(idx, 0.0)
        br = bm25_rank.get(idx, 10 ** 9)
        if sim >= SIM_THRESHOLD or br < BM25_TOPN:
            gated.append({"idx": idx, "sim": sim, "bm25_rank": br, "rrf": rrf,
                          "source": _metadata[idx]["source"],
                          "heading": _metadata[idx].get("heading", ""),
                          "text": _metadata[idx]["text"]})
        if len(gated) >= max(pool, k * 3):
            break

    if not gated:
        return []

    # реранкер поверх топ-кандидатов
    if RERANK_ON and len(gated) > 1:
        docs = [g["text"] for g in gated]
        rer = _rerank(query, docs, top_n=max(k, 3))
        if rer is not None:
            reranked = []
            max_score = rer[0][1] if rer else 0.0
            for doc_idx, score in rer:
                if score < RERANK_MIN_SCORE and score != max_score:
                    continue
                g = gated[doc_idx]
                g["rerank_score"] = score
                reranked.append(g)
            if reranked:
                return _diverse_select(reranked, k)
            # реранкер вернул пусто/все ниже порога → «не найдено»
            return []
    # без реранкера — порядок RRF + диверсификация
    gated.sort(key=lambda g: -g["rrf"])
    return _diverse_select(gated, k)


def _diverse_select(candidates: list[dict], k: int, max_per_source: int = 2) -> list[dict]:
    """
    MMR-подобная диверсификация: выбираем k результатов так, чтобы:
    - не более max_per_source чанков из одного файла;
    - чем больше разных источников, тем лучше;
    - если после диверсификации набралось меньше k, добираем лучших оставшихся
      кандидатов без ограничения по источнику, чтобы top_k гарантированно
      попадало в контекст LLM.
    """
    selected: list[dict] = []
    source_counts: dict[str, int] = {}
    # сортируем по убыванию релевантности (rerank_score > rrf > sim)
    sorted_cands = sorted(
        candidates,
        key=lambda g: -(g.get("rerank_score") or g.get("rrf") or g.get("sim", 0)),
    )
    for g in sorted_cands:
        src = g["source"]
        if source_counts.get(src, 0) >= max_per_source:
            continue
        selected.append(g)
        source_counts[src] = source_counts.get(src, 0) + 1
        if len(selected) >= k:
            return selected

    # fallback: если разных источников не хватило, добираем лучших оставшихся
    selected_ids = {id(g) for g in selected}
    for g in sorted_cands:
        if id(g) in selected_ids:
            continue
        selected.append(g)
        if len(selected) >= k:
            break
    return selected


def query(question: str, k: int = TOP_K, history: list[dict] | None = None) -> dict:
    """RAG-ответ по базе. Возвращает {answer, sources, empty, not_found, method}.

    Args:
        question: Текущий вопрос пользователя.
        k: Количество релевантных чанков для контекста.
        history: Список сообщений диалога вида [{"role": "user"|"assistant", "content": str}].
                 Используется для уточняющих вопросов.
    """
    if not RAG_AVAILABLE:
        return {"answer": "RAG-модуль недоступен.", "sources": [], "empty": True,
                "not_found": False, "method": "none"}
    if _index is None:
        _load_store()
    if _index is None or _index.ntotal == 0:
        return {"answer": "📚 База ВЭД пуста. Сначала выполните /ingest.",
                "sources": [], "empty": True, "not_found": False, "method": "empty"}

    q = (question or "").strip()
    if not q:
        return {"answer": "Задайте вопрос по ВЭД.", "sources": [], "empty": False,
                "not_found": False, "method": "noop"}

    try:
        hits = _hybrid_retrieve(q, k)
    except Exception as e:
        logger.error(f"RAG retrieve error: {e}", exc_info=True)
        return {"answer": f"⏱️/❌ Ошибка поиска: {e}", "sources": [], "empty": False,
                "not_found": False, "method": "error"}

    method = "rrf" if not (RERANK_ON and hits and "rerank_score" in hits[0]) else "rerank"
    if not hits:
        return {
            "answer": "По имеющейся базе ответа не нашёл. Уточните вопрос или переиндексируйте документы (/ingest).",
            "sources": [], "empty": False, "not_found": True, "method": method,
        }

    # сборка контекста
    context_parts: list[str] = []
    total = 0
    used_sources: dict[str, dict] = {}
    for g in hits:
        h = g["heading"]
        cite = f"[источник: {g['source']}" + (f", {h}" if h else "") + "]"
        block = f"{cite}\n{g['text']}"
        if total + len(block) > MAX_CONTEXT_LENGTH:
            break
        context_parts.append(block)
        total += len(block)
        # Источник идентифицируем по файлу + заголовку/сниппету, чтобы разные чанки
        # из одного файла учитывались отдельно при подсчёте top_k.
        src_key = f"{g['source']}#{h or g['text'][:60].strip().replace(chr(10), ' ')}"
        if src_key not in used_sources:
            used_sources[src_key] = {
                "source": g["source"], "heading": h,
                "chunk_id": g.get("chunk_id", 0),
                "snippet": g["text"][:200].replace("\n", " ").strip(),
                "sim": round(g.get("sim", 0.0), 3),
            }

    context = "\n\n---\n".join(context_parts)

    # сборка истории диалога
    history_block = ""
    if history:
        lines = []
        for msg in history[-6:]:  # используем последние 6 сообщений
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            label = "Пользователь" if role == "user" else "Ассистент" if role == "assistant" else role.capitalize()
            lines.append(f"{label}: {content}")
        if lines:
            history_block = "История диалога:\n" + "\n".join(lines) + "\n\n"

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, history=history_block, query=q)

    client = _get_client()

    def call():
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": prompt})
        return client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2, max_tokens=2000,
        )

    try:
        resp = _with_retries(call)
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"RAG query chat error: {e}", exc_info=True)
        return {"answer": f"❌ Ошибка генерации ответа: {e}", "sources": [],
                "empty": False, "not_found": False, "method": method}

    if not answer:
        answer = "По имеющейся базе ответа не нашёл."

    # Self-check: убеждаемся, что ответ опирается на предоставленный контекст.
    if ANSWER_VERIFY and answer and "не нашёл" not in answer.lower():
        grounded = _verify_answer_grounded(q, context, answer)
        if not grounded:
            logger.info("RAG answer failed self-check; returning not_found")
            return {
                "answer": "По имеющейся базе ответа не нашёл. Уточните вопрос или переиндексируйте документы (/ingest).",
                "sources": [], "empty": False, "not_found": True, "method": method,
            }

    return {"answer": answer, "sources": list(used_sources.values())[:5],
            "empty": False, "not_found": False, "method": method}


def _verify_answer_grounded(question: str, context: str, answer: str) -> bool:
    """
    Проверяет, основан ли ответ на предоставленном контексте.
    Возвращает True, если ответ grounded, иначе False.
    """
    if not _client:
        return True  # не блокируем ответ, если клиент недоступен
    verify_prompt = (
        "Оцени, основан ли ответ на предоставленном контексте.\n\n"
        f"Вопрос: {question}\n\n"
        f"Контекст:\n{context[:2000]}\n\n"
        f"Ответ:\n{answer[:1000]}\n\n"
        "Ответь только одним числом от 0 до 1, где 1 — полностью основан на контексте, "
        "0 — полностью не основан."
    )
    try:
        resp = _client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Ты строгий проверяющий фактов."},
                {"role": "user", "content": verify_prompt},
            ],
            temperature=0.0, max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip()
        score_match = re.search(r"[\d.,]+", raw.replace(",", "."))
        score = float(score_match.group()) if score_match else 0.5
        logger.info(f"Answer grounded score: {score}")
        return score >= VERIFY_MIN_GROUNDED_SCORE
    except Exception as e:
        logger.error(f"Answer verification error: {e}")
        return True


# ────────────────────────── публичный API: diagnostic ──────────────────────────
def diagnostic() -> dict:
    d = {"available": RAG_AVAILABLE, "faiss": _FAISS_OK, "openai": _OPENAI_OK,
         "bm25": _BM25_OK, "requests": _REQUESTS_OK, "api_key": bool(API_KEY),
         "base_url": BASE_URL, "embed_model": EMBED_MODEL, "chat_model": CHAT_MODEL,
         "rerank_on": RERANK_ON, "rerank_model": RERANK_MODEL, "rerank_url": RERANK_URL,
         "docs_dir": str(DOCS_DIR), "docs_count": 0, "indexed_chunks": 0}
    try:
        if DOCS_DIR.exists():
            d["docs_count"] = sum(1 for p in DOCS_DIR.iterdir()
                                  if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"})
    except Exception:
        pass
    d["indexed_chunks"] = int(_index.ntotal) if _index is not None else 0
    return d


# ────────────────────────── мини eval-сет (16 вопросов ВЭД) ──────────────────────────
# expected_sources — файлы, из которых должен прийти релевантный чанк.
# expected_in_answer — ключевые слова, которых ждем в ответе (хотя бы половина).
# expect_not_found — вопрос вне базы: ждём честный «не нашёл».
EVAL_SET: list[dict] = [
    {"q": "Какие справки нужны для валютного контроля по контракту на 5 млн рублей?",
     "expected_sources": ["fz_173_currency.txt"], "expected_in_answer": ["справка", "валютн"]},
    {"q": "Что такое паспорт сделки и в каких случаях он оформляется?",
     "expected_sources": ["fz_173_currency.txt"], "expected_in_answer": ["паспорт сделки", "контракт"]},
    {"q": "Что такое репатриация валюты и кто обязан её обеспечить?",
     "expected_sources": ["fz_173_currency.txt"], "expected_in_answer": ["репатриа", "валют"]},
    {"q": "В какие сроки подаётся таможенная декларация?",
     "expected_sources": ["tk_eaes.txt", "fz_289_customs.txt"], "expected_in_answer": ["срок", "деклар"]},
    {"q": "Какие таможенные режимы предусмотрены в ЕАЭС?",
     "expected_sources": ["tk_eaes.txt"], "expected_in_answer": ["режим", "ЕАЭС"]},
    {"q": "Какие документы нужны для таможенного оформления импорта товаров?",
     "expected_sources": ["fz_289_customs.txt", "tk_eaes.txt"], "expected_in_answer": ["документ", "деклар"]},
    {"q": "Что такое код ТН ВЭД и из скольких знаков он состоит?",
     "expected_sources": ["tn_ved_structure.txt"], "expected_in_answer": ["ТН ВЭД", "знак"]},
    {"q": "Какой орган принимает решения по таможенному регулированию в ЕАЭС?",
     "expected_sources": ["eec_decisions_key.txt"], "expected_in_answer": ["ЕЭК", "ЕАЭС"]},
    {"q": "Что означает термин EXW по Инкотермс 2020?",
     "expected_sources": ["incoterms_2020.txt"], "expected_in_answer": ["EXW", "франко"]},
    {"q": "Чем отличается FCA от FOB по Инкотермс 2020?",
     "expected_sources": ["incoterms_2020.txt"], "expected_in_answer": ["FCA", "FOB"]},
    {"q": "Что означает DDP в терминах Инкотермс 2020?",
     "expected_sources": ["incoterms_2020.txt"], "expected_in_answer": ["DDP", "пошлин"]},
    {"q": "Как определяется таможенная стоимость товаров?",
     "expected_sources": ["tk_eaes.txt", "fz_289_customs.txt"], "expected_in_answer": ["стоимост", "таможен"]},
    {"q": "Какие сведения должна содержать таможенная декларация?",
     "expected_sources": ["tk_eaes.txt", "fz_289_customs.txt"], "expected_in_answer": ["сведени", "деклар"]},
    {"q": "Что такое выпуск товаров и в какие сроки он производится?",
     "expected_sources": ["tk_eaes.txt", "fz_289_customs.txt"], "expected_in_answer": ["выпуск", "срок"]},
    {"q": "Какая ответственность за невыполнение обязанности по репатриации валюты?",
     "expected_sources": ["fz_173_currency.txt"], "expected_in_answer": ["ответствен", "валют"]},
    {"q": "Как приготовить борщ по рецепту Юлии Высоцкой?",
     "expected_sources": [], "expected_in_answer": [], "expect_not_found": True},
]


def _eval_check(item: dict, r: dict) -> tuple[bool, dict]:
    if item.get("expect_not_found"):
        ok = bool(r.get("not_found") or r.get("empty"))
        return ok, {"ok": ok, "reason": "expect_not_found", "got_not_found": r.get("not_found"),
                    "got_empty": r.get("empty")}
    sources = {s["source"] for s in r.get("sources", [])}
    exp_src = set(item.get("expected_sources", []))
    retrieval_ok = bool(sources & exp_src) if exp_src else True
    answer = (r.get("answer") or "").lower()
    kw = item.get("expected_in_answer", [])
    hits = [k for k in kw if k.lower() in answer]
    kw_hit = (len(hits) / len(kw)) if kw else 1.0
    kw_ok = kw_hit >= 0.5
    ok = retrieval_ok and kw_ok
    return ok, {"ok": ok, "retrieval_ok": retrieval_ok,
                "sources_got": sorted(sources), "kw_hits": hits, "kw_hit": round(kw_hit, 2),
                "not_found": r.get("not_found"), "method": r.get("method")}


def run_eval() -> dict:
    if not RAG_AVAILABLE:
        return {"available": False, "error": "RAG недоступен"}
    if _index is None:
        _load_store()
    if _index is None or _index.ntotal == 0:
        return {"available": True, "error": "База пуста — сначала /ingest", "results": []}

    per_q: list[dict] = []
    n_pass = 0
    retrieval_pass = 0
    kw_sum = 0.0
    notfound_total = 0
    notfound_ok = 0
    for item in EVAL_SET:
        r = query(item["q"])
        ok, detail = _eval_check(item, r)
        per_q.append({"q": item["q"], **detail, "answer_preview": (r.get("answer") or "")[:160]})
        n_pass += int(ok)
        if not item.get("expect_not_found"):
            retrieval_pass += int(detail["retrieval_ok"])
            kw_sum += detail["kw_hit"]
        else:
            notfound_total += 1
            notfound_ok += int(ok)

    total = len(EVAL_SET)
    graded = total - notfound_total
    summary = {
        "pass": n_pass, "total": total, "pass_rate": round(n_pass / total, 3),
        "retrieval_recall": round(retrieval_pass / graded, 3) if graded else None,
        "avg_kw_hit": round(kw_sum / graded, 3) if graded else None,
        "notfound_correct": f"{notfound_ok}/{notfound_total}",
    }
    report = {"summary": summary, "results": per_q,
              "stats": get_stats(), "config": {"top_k": TOP_K, "sim_threshold": SIM_THRESHOLD,
                                                "bm25_topn": BM25_TOPN, "candidate_pool": CANDIDATE_POOL,
                                                "rerank_on": RERANK_ON, "rerank_model": RERANK_MODEL}}
    try:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        EVAL_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"eval report не сохранён: {e}")
    return report


# ────────────────────────── CLI ──────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "ingest":
        print(json.dumps(ingest(verbose=True), ensure_ascii=False, indent=2))
    elif cmd == "stats":
        print(json.dumps(get_stats(), ensure_ascii=False, indent=2))
    elif cmd == "diag":
        print(json.dumps(diagnostic(), ensure_ascii=False, indent=2))
    elif cmd == "eval":
        print(json.dumps(run_eval(), ensure_ascii=False, indent=2))
    elif cmd == "query":
        q = " ".join(sys.argv[2:])
        print(json.dumps(query(q), ensure_ascii=False, indent=2) if q
              else "укажите вопрос: rag_engine.py query <вопрос>")
    else:
        print("команды: stats | diag | ingest | eval | query <вопрос>")