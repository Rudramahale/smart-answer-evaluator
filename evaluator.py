"""
Semantic scoring: bi-encoder (all-mpnet-base-v2) plus optional cross-encoder
(reference vs student) for sharper paraphrase judgments on borderline answers.

Environment (optional):
  ANSWER_EVAL_USE_CROSS_ENCODER   default "1"; set "0"/"false"/"off" to skip (faster).
  ANSWER_EVAL_CROSS_ENCODER_MODEL default "cross-encoder/stsb-distilroberta-base"
                                  (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 uses sigmoid).
  ANSWER_EVAL_CE_FASTPATH         default "1"; when bi-encoder is very confident, skip the
                                  cross-encoder forward pass (same marks in practice, lower latency).
"""
import csv
import os
import pickle
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from preprocess import AI_KEYWORD_SYNONYMS, preprocess

# Cross-encoder (optional): set ANSWER_EVAL_USE_CROSS_ENCODER=0 to disable.
_CROSS_ENCODER = None
_CROSS_ENCODER_MODEL = os.environ.get(
    "ANSWER_EVAL_CROSS_ENCODER_MODEL",
    "cross-encoder/stsb-distilroberta-base",
)

print("Loading model...")
model = SentenceTransformer("all-mpnet-base-v2")

print("Loading dataset...")
_dataset_path = "new_dataset.csv"
if not os.path.isfile(_dataset_path):
    raise RuntimeError("new_dataset.csv not found.")

with open(_dataset_path, newline="", encoding="utf-8") as _f:
    _csv_rows = list(csv.DictReader(_f))

_rows_by_id = {int(r["ID"]): r for r in _csv_rows}

# Filled by _populate_rubric_caches() — rubric keywords preprocessed, regex compiled, ref text truncated once.
_KEYWORD_ENTRIES_BY_QID = {}
_REF_TRUNC_BY_QID = {}

EMBEDDINGS_DIR = "embeddings"
# New cache: normalized expected + QA-context vectors (see semantic_score).
CACHE_FILE = os.path.join(EMBEDDINGS_DIR, "cached_qa_embeddings.pkl")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

expected_embeddings_cache = {}
context_embeddings_cache = {}


def _load_or_build_caches():
    global expected_embeddings_cache, context_embeddings_cache
    if os.path.exists(CACHE_FILE):
        print("Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as f:
            bundle = pickle.load(f)
        if (
            isinstance(bundle, dict)
            and bundle.get("expected")
            and bundle.get("context")
        ):
            expected_embeddings_cache = bundle["expected"]
            context_embeddings_cache = bundle["context"]
            return

    print("Generating embeddings (normalized, question+answer context)...")
    expected_embeddings_cache = {}
    context_embeddings_cache = {}
    for row in _csv_rows:
        qid = int(row["ID"])
        answer_text = str(row["Answer"]).replace("\n", " ").strip()
        question_text = str(row["Question"]).strip()
        ctx_text = f"{question_text} {answer_text}"
        expected_embeddings_cache[qid] = model.encode(
            [answer_text], normalize_embeddings=True
        )
        context_embeddings_cache[qid] = model.encode(
            [ctx_text], normalize_embeddings=True
        )
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(
            {"expected": expected_embeddings_cache, "context": context_embeddings_cache},
            f,
        )
    print("Embeddings cached.")


_load_or_build_caches()


def _use_cross_encoder() -> bool:
    v = os.environ.get("ANSWER_EVAL_USE_CROSS_ENCODER", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _init_cross_encoder():
    global _CROSS_ENCODER
    if not _use_cross_encoder():
        print("Cross-encoder: disabled (ANSWER_EVAL_USE_CROSS_ENCODER).")
        return
    try:
        from sentence_transformers import CrossEncoder

        print(f"Loading cross-encoder ({_CROSS_ENCODER_MODEL})...")
        _CROSS_ENCODER = CrossEncoder(_CROSS_ENCODER_MODEL, max_length=512)
        print("Cross-encoder ready.")
    except Exception as e:
        print(f"Cross-encoder unavailable, using bi-encoder only: {e}")
        _CROSS_ENCODER = None


_init_cross_encoder()

# SCORING FUNCTIONS

def _lemma_token_set(processed_text: str) -> set:
    return {w for w in processed_text.split() if w}


def _keyword_entry_for_rubric_term(raw_kw: str):
    """Precompute one rubric keyword: token set + synonym token sets + substring regex."""
    raw_kw = raw_kw.strip()
    if not raw_kw:
        return None, None
    kw_lower = raw_kw.lower()
    pat = (
        re.compile(r"\b" + re.escape(kw_lower) + r"\b")
        if len(kw_lower) >= 2
        else None
    )
    kw_proc = preprocess(raw_kw)
    kw_tokens = frozenset(_lemma_token_set(kw_proc))
    if not kw_tokens:
        return None, pat
    syn_sets = []
    for tok in kw_tokens:
        for syn in AI_KEYWORD_SYNONYMS.get(tok, ()):
            st = frozenset(_lemma_token_set(preprocess(syn)))
            if st:
                syn_sets.append(st)
    return {"kw_tokens": kw_tokens, "syn_sets": tuple(syn_sets)}, pat


def count_matched_keywords_from_entries(processed_answer: str, entries: tuple) -> int:
    answer_tokens = _lemma_token_set(processed_answer)
    if not answer_tokens:
        return 0
    matched = 0
    for spec, _ in entries:
        if spec is None:
            continue
        kw_tokens = spec["kw_tokens"]
        if kw_tokens <= answer_tokens:
            matched += 1
            continue
        if len(kw_tokens) > 1:
            overlap_ratio = len(kw_tokens & answer_tokens) / len(kw_tokens)
            if overlap_ratio >= 0.65:
                matched += 1
                continue
        syn_hit = False
        for syn_tokens in spec["syn_sets"]:
            if syn_tokens and syn_tokens <= answer_tokens:
                syn_hit = True
                break
        if syn_hit:
            matched += 1
    return matched


def _substring_fallback_from_entries(text_lower: str, entries: tuple) -> bool:
    for _, pat in entries:
        if pat is not None and pat.search(text_lower):
            return True
    return False


def _populate_rubric_caches():
    global _KEYWORD_ENTRIES_BY_QID, _REF_TRUNC_BY_QID
    for row in _csv_rows:
        qid = int(row["ID"])
        ks = str(row["Keywords"])
        kws = ks.split(",") if ks != "nan" else []
        entries = []
        for raw_kw in kws:
            spec, pat = _keyword_entry_for_rubric_term(raw_kw)
            entries.append((spec, pat))
        _KEYWORD_ENTRIES_BY_QID[qid] = tuple(entries)
        _REF_TRUNC_BY_QID[qid] = str(row["Answer"]).replace("\n", " ").strip()[:3200]
    print(
        f"Rubric keyword/reference caches built for {len(_KEYWORD_ENTRIES_BY_QID)} questions."
    )


_populate_rubric_caches()


def count_matched_keywords(processed_answer: str, keywords: list) -> int:
    """
    Legacy path: count from raw keyword strings (slower). Prefer entries cache in check_answer.
    """
    answer_tokens = _lemma_token_set(processed_answer)
    if not answer_tokens:
        return 0

    matched = 0
    for raw_kw in keywords:
        raw_kw = raw_kw.strip()
        if not raw_kw:
            continue
        kw_proc = preprocess(raw_kw)
        kw_tokens = _lemma_token_set(kw_proc)
        if not kw_tokens:
            continue

        if kw_tokens <= answer_tokens:
            matched += 1
            continue

        if len(kw_tokens) > 1:
            overlap_ratio = len(kw_tokens & answer_tokens) / len(kw_tokens)
            if overlap_ratio >= 0.65:
                matched += 1
                continue

        synonym_hit = False
        for tok in kw_tokens:
            for syn in AI_KEYWORD_SYNONYMS.get(tok, ()):
                syn_tokens = _lemma_token_set(preprocess(syn))
                if syn_tokens and syn_tokens <= answer_tokens:
                    synonym_hit = True
                    break
            if synonym_hit:
                break
        if synonym_hit:
            matched += 1

    return matched


def _has_keyword_substring_fallback(student_raw: str, keywords: list) -> bool:
    """Used only when rubric entry cache is unavailable."""
    text = student_raw.lower()
    for raw_kw in keywords:
        kw = raw_kw.strip().lower()
        if len(kw) < 2:
            continue
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            return True
    return False


def semantic_scores_from_vec(q_id: int, student_vec: np.ndarray):
    """
    Returns (blended_semantic_score, sim_to_expected_only).
    student_vec must be L2-normalized (same as encode(..., normalize_embeddings=True)).
    """
    emb_exp = expected_embeddings_cache.get(q_id)
    emb_ctx = context_embeddings_cache.get(q_id)
    if emb_exp is None or emb_ctx is None:
        return 0.0, 0.0
    sim_exp = float(np.dot(emb_exp[0], student_vec))
    sim_ctx = float(np.dot(emb_ctx[0], student_vec))
    combined = 0.64 * sim_exp + 0.36 * sim_ctx
    return float(np.clip(combined, 0.0, 1.0)), sim_exp


def _cross_encoder_score_to_unit(raw: float) -> float:
    """Map cross-encoder raw output to [0, 1] for fusion with bi-encoder cosine."""
    if "stsb" in _CROSS_ENCODER_MODEL.lower():
        return float(np.clip(raw / 5.0, 0.0, 1.0))
    return float(np.clip(1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0))), 0.0, 1.0))


def _cross_encoder_fastpath(s_bi: float, sim_exp: float) -> bool:
    """Skip CE when bi-encoder is already very aligned (saves one transformer forward)."""
    v = os.environ.get("ANSWER_EVAL_CE_FASTPATH", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return s_bi >= 0.60 and sim_exp >= 0.53


def fused_semantic_score(
    question_id: int,
    student_answer: str,
    student_vec: np.ndarray,
    reference_answer: str,
):
    """
    Bi-encoder cosine blend plus optional cross-encoder (reference vs student).
    Returns (fused_score, sim_to_expected_only); fused_score drives semantic bands.
    """
    s_bi, sim_exp = semantic_scores_from_vec(question_id, student_vec)
    if _CROSS_ENCODER is None:
        return s_bi, sim_exp
    if _cross_encoder_fastpath(s_bi, sim_exp):
        return s_bi, sim_exp

    ref = _REF_TRUNC_BY_QID.get(question_id)
    if not ref:
        ref = str(reference_answer).replace("\n", " ").strip()[:3200]
    stu = student_answer.strip()[:3200]
    if not ref or not stu:
        return s_bi, sim_exp

    raw = float(
        _CROSS_ENCODER.predict([(ref, stu)], show_progress_bar=False)[0]
    )
    s_ce = _cross_encoder_score_to_unit(raw)
    fused = float(np.clip(0.34 * s_bi + 0.66 * s_ce, 0.0, 1.0))
    return fused, sim_exp



# MAIN EVALUATION FUNCTION

def check_answer(question_id: int, student_answer: str) -> dict:
    """
    Evaluate a student's answer and return marks out of 6.
    """
    row = _rows_by_id.get(int(question_id))
    if row is None:
        return {"error": "Question ID not found"}
    if not student_answer or not student_answer.strip():
        return {"error": "Empty answer provided"}

    raw_words = len(student_answer.split())

    # Gate 1: Minimum raw word count
    if raw_words < 5:
        return {"marks": 0}

    # Gate 2: Minimum meaningful words after preprocessing
    processed_answer = preprocess(student_answer)
    meaningful_count = len(processed_answer.split())
    if meaningful_count < 4:
        return {"marks": 0}

    # Load keywords (strings for legacy path) + precomputed rubric entries (fast path)
    keywords_str = str(row["Keywords"])
    keywords = keywords_str.split(",") if keywords_str != "nan" else []
    entries = _KEYWORD_ENTRIES_BY_QID.get(int(question_id), ())
    student_lower = student_answer.lower()

    # Gate 3: Lexical keyword count; substring fallback only passes the gate at 1 match
    if entries:
        matched_keywords = count_matched_keywords_from_entries(processed_answer, entries)
        if matched_keywords == 0 and _substring_fallback_from_entries(
            student_lower, entries
        ):
            matched_keywords = 1
    else:
        matched_keywords = count_matched_keywords(processed_answer, keywords)
        if matched_keywords == 0 and _has_keyword_substring_fallback(
            student_answer, keywords
        ):
            matched_keywords = 1
    if matched_keywords == 0:
        return {"marks": 0}

    student_vec = model.encode([student_answer], normalize_embeddings=True)[0]
    reference_answer = str(row["Answer"])
    s_score, sim_exp = fused_semantic_score(
        question_id, student_answer, student_vec, reference_answer
    )

    # Gate 4: Off-topic / nonsense
    if s_score < 0.20:
        return {"marks": 0}
    if sim_exp < 0.14 and s_score < 0.28:
        return {"marks": 0}

    # Semantic bands (calibrated for bi + cross-encoder fusion)
    #   good  -> s_score >= 0.48
    #   avg   -> 0.33 <= s_score < 0.48
    good_sem = s_score >= 0.48
    avg_sem = 0.33 <= s_score < 0.48

    if matched_keywords >= 5:
        marks = 6 if (good_sem or avg_sem) else 5
    elif matched_keywords == 4:
        marks = 6 if good_sem else (5 if avg_sem else 4)
    elif matched_keywords == 3:
        marks = 5 if good_sem else (4 if avg_sem else 3)
    elif matched_keywords == 2:
        marks = 4 if good_sem else (3 if avg_sem else 2)
    elif matched_keywords == 1:
        marks = 2 if good_sem else (1 if avg_sem else 0)
    else:
        marks = 0

    # Depth cap: raw length (substance vs keyword lists)
    if raw_words < 14:
        marks = min(marks, 2)
    elif raw_words < 26:
        marks = min(marks, 4)

    keyword_density = matched_keywords / max(meaningful_count, 1)
    if keyword_density > 0.40 and raw_words < 22:
        marks = min(marks, 2)

    # High marks need multiple keyword evidence; do not cap 2-keyword answers at 2 (old bug)
    if matched_keywords <= 1 and marks > 2:
        marks = min(marks, 2)

    marks = max(0, min(6, marks))
    return {"marks": marks}
