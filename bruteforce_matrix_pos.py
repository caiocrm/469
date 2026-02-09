from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import random
import re
import signal
import string
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BASE_MATRIX_4x4 = [
    [3, 1, 6, 1],
    [1, 2, 1, 1],
    [1, 1, 3, 1],
    [4, 6, 1, 1],
]

BOOK1_ROWS = [
    "dtjfhg",
    "jhfvzk",
    "bbliiug",
    "bkjjjjjjj",
    "xhvuo",
    "fffff",
    "zkkbk h",
    "lbhiovz",
    "klhi igbb",
]

BOOK2_ROWS = [
    ["ljkhbl", "nilse",   "jfpce",   "ojvco",   "ld"],
    ["slcld",  "ylddiv",  "dnolsd",  "dd",      "sd"],
    ["sdcp",   "cppcs",   "cccpc",   "cpsc"],
    ["awdp",   "cpcw",    "cfw",     "ce"],
    ["cpvc",   "ev",      "vcemmev", "vrvf"],
    ["cp",     "fd",      "vmfpm",   "xcv"],
]

BOOK2_FLAT = [b for row in BOOK2_ROWS for b in row]

BOOK2_ORIGINAL_TEXT = """ljkhbl nilse jfpce ojvco ld
slcld ylddiv dnolsd dd sd
sdcp cppcs cccpc cpsc
awdp cpcw cfw ce
cpvc ev vcemmev vrvf
cp fd vmfpm xcv
"""
_BOOK2_STRIPPED_ORIG = re.sub(r"[^a-zA-Z]", "", BOOK2_ORIGINAL_TEXT).lower()
BOOK2_TETRAGRAMS = [_BOOK2_STRIPPED_ORIG[i:i+4] for i in range(0, len(_BOOK2_STRIPPED_ORIG), 4)]
assert len(BOOK2_TETRAGRAMS) == 26 and all(len(t) == 4 for t in BOOK2_TETRAGRAMS), "Book2 tetragram parse failed"

AZ = string.ascii_uppercase
AZ_SET = set(AZ)

HARD_CFG = {
    "family": "matrix_book2",
    "token_w": 4,
    "offset": 3,
    "use_boundaries": True,
    "coord_mode": "ab",
    "row_map": "v-1",
    "col_rule": "d-c",
    "book2_mode": "rows",
}

DEFAULT_SOURCES = [
    "89521972781670512164",
    "6249684756019965855064996704672611458",
    "69042204648451911889521977128895219594576552364672119118",
    "64671180014015255175191180189445229534527477090967347092012857197278304042158576594344215956042183",
    "3464651800996734057928275857651972788943887215128895219618",
    "11457278572611857642197096805796366125275705845217652197278304648765159564611414519889975112161518005953724348562510811463646724353451586042158577445451904504215956151353478019288952160199364672431427894315191186512819118",
    "5611472611646713646461219785857651972921972781670546711800140152551751911801894452295345274464726114514519485611451908304576512282177350843485628477090889524348561121648006586756356114519199118",
    "6468895219911036512889672127788943887215128895219618",
    "11457278572611857642197096805796366125275705845217652197278304648765159564611414519889975112161518005458561197353646724348561145196726114519",
    "80065861143128895",
    "57576546",
    "64671180014015255175191180189445229534527446472611451451948561145190830457651228217735084348562756356114519199118",
    "64688952199118435081243485611451912112888304646797278316017464834943435452197212777090967347092012857197278304042158576594344215956042183",
    "03118065719189434",
    "0421585765121615180094343508434856114572785726118576436467243534527560192889521973536467249684756019968477090889521972781670512164856114519199118",
    "6468895219911800651288952364672119118",
    "5765135347830464679727839673405792827585765125275705845217652197278304648765159564611414519889975159537243485612783",
    "3478",
    "51353478",
]


def append_jsonl(path: str, obj: Dict[str, Any], lock_path: Optional[str] = None) -> None:
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    if lock_path:
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(0.005)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        finally:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

def rotate_matrix_cw(mat: List[List[int]], k90: int) -> List[List[int]]:
    k90 %= 4
    m = [row[:] for row in mat]
    for _ in range(k90):
        m = [list(row) for row in zip(*m[::-1])]
    return m

def permute_rows(mat: List[List[int]], perm: Tuple[int, ...]) -> List[List[int]]:
    return [mat[i][:] for i in perm]

def permute_cols(mat: List[List[int]], perm: Tuple[int, ...]) -> List[List[int]]:
    return [[row[j] for j in perm] for row in mat]

def iter_matrix_variants(mode: str) -> Iterable[Tuple[str, List[List[int]]]]:
    rots = [(f"rot{k}", rotate_matrix_cw(BASE_MATRIX_4x4, k)) for k in range(4)]
    if mode == "rot":
        yield from rots
        return

    perms24 = list(itertools.permutations(range(4)))

    def lite_perms() -> List[Tuple[int, ...]]:
        candidates = set()
        candidates.add((0,1,2,3))
        candidates.add((3,2,1,0))
        candidates.add((1,0,2,3))
        candidates.add((0,2,1,3))
        candidates.add((0,1,3,2))
        candidates.add((2,3,0,1))
        candidates.add((1,2,3,0))
        candidates.add((3,0,1,2))
        return list(candidates)

    if mode == "rot_rows_cols_lite":
        rp = lite_perms()
        cp = lite_perms()
        for rname, rm in rots:
            for rperm in rp:
                mr = permute_rows(rm, rperm)
                for cperm in cp:
                    mc = permute_cols(mr, cperm)
                    yield (f"{rname}_r{rperm}_c{cperm}", mc)
        return

    if mode in {"rot_rows", "rot_cols", "rot_rows_cols"}:
        for rname, rm in rots:
            if mode == "rot_rows":
                for rperm in perms24:
                    yield (f"{rname}_r{rperm}", permute_rows(rm, rperm))
            elif mode == "rot_cols":
                for cperm in perms24:
                    yield (f"{rname}_c{cperm}", permute_cols(rm, cperm))
            else:
                for rperm in perms24:
                    mr = permute_rows(rm, rperm)
                    for cperm in perms24:
                        yield (f"{rname}_r{rperm}_c{cperm}", permute_cols(mr, cperm))
        return

    raise ValueError(f"Unknown matrix mode: {mode}")

def build_matrix_map(mode: str) -> Dict[str, List[List[int]]]:
    return {name: mat for name, mat in iter_matrix_variants(mode)}

@dataclass(frozen=True)
class Source:
    name: str
    raw: str

def normalize_digits_keep_boundaries(s: str) -> Tuple[str, List[int]]:
    digits: List[str] = []
    boundaries: List[int] = []
    prev_was_digit = False
    for ch in s:
        if ch.isdigit():
            digits.append(ch)
            prev_was_digit = True
        else:
            if prev_was_digit:
                boundaries.append(len(digits))
            prev_was_digit = False
    boundaries = sorted(set(b for b in boundaries if 0 < b < len(digits)))
    return "".join(digits), boundaries

def chunk_by_fixed_width(digits: str, width: int, offset: int = 0, stop_at_boundaries: Optional[List[int]] = None) -> List[str]:
    if width <= 0:
        return []
    out: List[str] = []
    i = offset
    boundary_set = set(stop_at_boundaries or [])
    L = len(digits)
    while i + width <= L:
        if stop_at_boundaries:
            internal = [b for b in boundary_set if i < b < i+width]
            if internal:
                i = min(internal)
                continue
        out.append(digits[i:i+width])
        i += width
    return out

def map_to_AZ(idx: int) -> str:
    return AZ[idx % 26]

def decode_family_matrix_book2(
    digits: str,
    boundaries: List[int],
    mat: List[List[int]],
    token_w: int,
    offset: int,
    use_boundaries: bool,
    coord_mode: str,
    row_map: str,
    col_rule: str,
    book2_mode: str,
) -> str:
    toks = chunk_by_fixed_width(digits, token_w, offset, boundaries if use_boundaries else None)
    out: List[str] = []

    for t in toks:
        if len(t) < 4:
            continue
        a, b, c, d = (ord(t[0]) - 48, ord(t[1]) - 48, ord(t[2]) - 48, ord(t[3]) - 48)

        if coord_mode == "ab":
            r, cc = a % 4, b % 4
        else:
            r, cc = b % 4, a % 4

        v = mat[r][cc]

        if row_map == "v-1":
            row_i = v - 1
        elif row_map == "v%6":
            row_i = v % 6
        else:
            row_i = (v - 1) % 6

        if book2_mode == "rows":
            row = BOOK2_ROWS[row_i]
            L = len(row)
            if L <= 0:
                continue

            if col_rule == "c-d":
                col = (c - d) % L
            elif col_rule == "d-c":
                col = (d - c) % L
            elif col_rule == "10c+d":
                col = (10 * c + d) % L
            elif col_rule == "c":
                col = c % L
            elif col_rule == "d":
                col = d % L
            elif col_rule == "a+b":
                col = (a + b) % L
            else:
                col = (b + a) % L

            block = row[col]
            try:
                idx = BOOK2_FLAT.index(block)
            except ValueError:
                idx = (sum(ord(x) for x in block) % 26)
            out.append(map_to_AZ(idx))
        else:
            if col_rule == "c-d":
                sel = (c - d)
            elif col_rule == "d-c":
                sel = (d - c)
            elif col_rule == "10c+d":
                sel = (10 * c + d)
            elif col_rule == "c":
                sel = c
            elif col_rule == "d":
                sel = d
            elif col_rule == "a+b":
                sel = a + b
            else:
                sel = b + a
            idx = (row_i * 9 + sel) % 26
            out.append(map_to_AZ(idx))

    return "".join(out)

COMMON_EN = {
    "THE","AND","THAT","THIS","WITH","YOU","YOUR","HAVE","NOT","FOR","ARE","WAS","BUT","HIS","HER","THEY",
    "ONE","ALL","CAN","WILL","FROM","WHAT","WHEN","WHERE","THERE","WHICH","WHO","WHY","HOW","SOME","MORE",
    "BE","TO","OF","IN","IS","IT","ON","AS","AT","BY","OR",
}
COMMON_DE = {
    "DER","DIE","DAS","UND","NICHT","ICH","DU","SIE","WIR","IHR","ES","EIN","EINE","MIT","FUR","FÜR","AUF",
    "IST","WAR","HAT","HABE","HABEN","WENN","WIE","WO","WAS","WER","WARUM","DASS","ABER","VON","ZU","IM","IN",
}

EN_BIGRAM_TOP = [
    "TH","HE","IN","ER","AN","RE","ON","AT","EN","ND","TI","ES","OR","TE","OF","ED","IS","IT","AL","AR",
    "ST","TO","NT","NG","SE","HA","AS","OU","IO","LE",
]
DE_BIGRAM_TOP = [
    "ER","EN","CH","DE","EI","TE","IN","ND","IE","GE","UN","ST","NE","BE","AU","SC","DI","HE","IC","SE",
    "ES","RE","LE","DA","AN","KE","IS","AS","NS","RA",
]

def make_bigram_weight(top: List[str]) -> Dict[str, float]:
    w = {}
    for i, bg in enumerate(top):
        w[bg] = 2.0 - (i / max(1, len(top)-1))
    return w

EN_BG_W = make_bigram_weight(EN_BIGRAM_TOP)
DE_BG_W = make_bigram_weight(DE_BIGRAM_TOP)

@dataclass
class ScoreBreakdown:
    length: int
    alpha_ratio: float
    vowel_ratio: float
    ioc: float
    run_penalty: float
    bigram_score: float
    word_hits: int
    total: float

VOWELS = set("AEIOUY")

def index_of_coincidence(text: str) -> float:
    counts = Counter(ch for ch in text if ch in AZ_SET)
    N = sum(counts.values())
    if N < 2:
        return 0.0
    num = sum(c * (c - 1) for c in counts.values())
    den = N * (N - 1)
    return num / den

def longest_run_penalty(text: str) -> float:
    if not text:
        return 0.0
    max_run = 1
    cur = 1
    for i in range(1, len(text)):
        if text[i] == text[i-1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return max(0.0, (max_run - 6) * 0.8)

def bigram_plausibility(text: str, bg_w: Dict[str, float]) -> float:
    t = "".join(ch for ch in text if ch in AZ_SET)
    if len(t) < 2:
        return 0.0
    score = 0.0
    unseen = 0
    for i in range(len(t) - 1):
        bg = t[i:i+2]
        w = bg_w.get(bg)
        if w is None:
            unseen += 1
        else:
            score += w
    score -= 0.15 * unseen
    return score / max(1, len(t) - 1)

def count_word_hits(text: str, words: set) -> int:
    t = text.upper()
    hits = 0
    for w in words:
        if len(w) < 2:
            continue
        hits += len(re.findall(re.escape(w), t))
    return hits



def score_text(text: str, lang: str) -> ScoreBreakdown:
    if not text:
        return ScoreBreakdown(0,0,0,0,0,0,0,0,-1e9)

    t = text.upper()
    letters = [ch for ch in t if ch in AZ_SET]
    L = len(t)
    if L == 0:
        return ScoreBreakdown(0,0,0,0,0,0,0,0,-1e9)

    alpha_ratio = (len(letters) / L) if L else 0.0
    vowel_ratio = (sum(1 for ch in letters if ch in VOWELS) / max(1, len(letters)))
    ioc = index_of_coincidence(t)
    run_pen = longest_run_penalty(t)

    if lang == "en":
        bg = bigram_plausibility(t, EN_BG_W)
        wh = count_word_hits(t, COMMON_EN)
    else:
        bg = bigram_plausibility(t, DE_BG_W)
        wh = count_word_hits(t, COMMON_DE)

    ioc_target = 0.062
    ioc_score = -abs(ioc - ioc_target) * 20.0

    vowel_target = 0.38
    vowel_score = -abs(vowel_ratio - vowel_target) * 8.0

    total = (
        3.0 * bg
        + 0.8 * wh
        + 2.0 * alpha_ratio
        + ioc_score
        + vowel_score
        - run_pen
    )

    return ScoreBreakdown(
        length=L,
        alpha_ratio=alpha_ratio,
        vowel_ratio=vowel_ratio,
        ioc=ioc,
        run_penalty=run_pen,
        bigram_score=bg,
        word_hits=wh,
        total=total
    )


@dataclass(frozen=True)
class MatrixOnlyConfig:
    matrix_variant: str

def config_hash(matrix_variant: str, mat: List[List[int]]) -> str:
    payload = {
        "matrix_variant": matrix_variant,
        "matrix": mat,
        "params": HARD_CFG,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

@dataclass
class EvalResult:
    matrix_variant: str
    matrix: List[List[int]]
    cfg_hash: str
    score_en: ScoreBreakdown
    score_de: ScoreBreakdown
    score_combo: float
    decoded_samples: Dict[str, str]
    decoded_full_len: Dict[str, int]
    notes: List[str]

def evaluate_matrix_on_sources(
    matrix_variant: str,
    mat: List[List[int]],
    sources: List[Source],
    sample_len: int = 240,
) -> EvalResult:
    decoded: Dict[str, str] = {}
    full_len: Dict[str, int] = {}
    notes: List[str] = []

    for src in sources:
        digits, bounds = normalize_digits_keep_boundaries(src.raw)

        text = decode_family_matrix_book2(
            digits, bounds, mat,
            token_w=HARD_CFG["token_w"],
            offset=HARD_CFG["offset"],
            use_boundaries=HARD_CFG["use_boundaries"],
            coord_mode=HARD_CFG["coord_mode"],
            row_map=HARD_CFG["row_map"],
            col_rule=HARD_CFG["col_rule"],
            book2_mode=HARD_CFG["book2_mode"],
        )

        decoded[src.name] = text[:sample_len]
        full_len[src.name] = len(text)

    joined = " ".join(decoded.values())
    se = score_text(joined, "en")
    sd = score_text(joined, "de")
    combo = 0.55 * se.total + 0.45 * sd.total

    return EvalResult(
        matrix_variant=matrix_variant,
        matrix=mat,
        cfg_hash=config_hash(matrix_variant, mat),
        score_en=se,
        score_de=sd,
        score_combo=combo,
        decoded_samples=decoded,
        decoded_full_len=full_len,
        notes=notes,
    )

def is_promising(res: EvalResult, threshold: float, must_have: Optional[List[str]] = None) -> bool:
    if res.score_combo < threshold:
        return False
    if must_have:
        U = " ".join(res.decoded_samples.values()).upper()
        for s in must_have:
            if s.upper() not in U:
                return False
    return True

def split_evenly(items: List[Any], n: int) -> List[List[Any]]:
    n = max(1, n)
    buckets = [[] for _ in range(n)]
    for i, it in enumerate(items):
        buckets[i % n].append(it)
    return buckets

def worker_run(
    worker_id: int,
    variants: List[str],
    sources_payload: List[Tuple[str, str]],
    matrix_map_payload: Dict[str, List[List[int]]],
    out_path: str,
    lock_path: Optional[str],
    threshold: float,
    must_have: Optional[List[str]],
    sample_len: int,
    report_every: int,
) -> Dict[str, Any]:
    sources = [Source(name=n, raw=r) for (n, r) in sources_payload]
    matrix_map = matrix_map_payload

    best: List[Tuple[float, Dict[str, Any]]] = []
    seen = 0

    for mv in variants:
        seen += 1
        try:
            mat = matrix_map[mv]
            res = evaluate_matrix_on_sources(mv, mat, sources, sample_len=sample_len)
        except Exception:
            continue

        if is_promising(res, threshold=threshold, must_have=must_have):
            record = {
                "ts": time.time(),
                "worker": worker_id,
                "cfg_hash": res.cfg_hash,
                "family": HARD_CFG["family"],
                "hardcoded_params": HARD_CFG,
                "matrix_variant": res.matrix_variant,
                "matrix": res.matrix,
                "score_combo": res.score_combo,
                "score_en": dataclasses.asdict(res.score_en),
                "score_de": dataclasses.asdict(res.score_de),
                "notes": res.notes,
                "decoded_samples": res.decoded_samples,
                "decoded_full_len": res.decoded_full_len,
            }
            append_jsonl(out_path, record, lock_path=lock_path)

            best.append((res.score_combo, record))
            best.sort(key=lambda x: x[0], reverse=True)
            best = best[:30]

        if report_every > 0 and (seen % report_every == 0):
            prog = {
                "ts": time.time(),
                "worker": worker_id,
                "progress": {"seen": seen, "total": len(variants)},
            }
            append_jsonl(out_path, prog, lock_path=lock_path)

    return {"worker": worker_id, "seen": seen, "top": [r for _, r in best]}

def parse_sources(args: argparse.Namespace) -> List[Source]:
    sources: List[Source] = []
    if args.sources_json:
        with open(args.sources_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list) and obj and isinstance(obj[0], str):
            for i, s in enumerate(obj):
                sources.append(Source(name=f"src{i}", raw=s))
        elif isinstance(obj, list):
            for it in obj:
                sources.append(Source(name=str(it.get("name", "src")), raw=str(it.get("raw", ""))))
        else:
            raise ValueError("sources_json must be a list")
    else:
        for i, s in enumerate(args.source or []):
            if ":" in s and s.split(":", 1)[0].isalnum():
                name, raw = s.split(":", 1)
                sources.append(Source(name=name, raw=raw))
            else:
                sources.append(Source(name=f"src{i}", raw=s))
    return sources

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-json", help="JSON file containing sources (list[str] or list[{name,raw}])")
    ap.add_argument("--source", action="append", help='Inline source. Either "name:digits..." or just digits. Can repeat.')
    ap.add_argument("--out", default="results_469.jsonl", help="Output JSONL file (appended).")
    ap.add_argument("--lock", default="results_469.lock", help="Lock file path (set empty to disable).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--matrix-mode", default="rot_rows_cols_lite",
                    choices=["rot","rot_rows","rot_cols","rot_rows_cols","rot_rows_cols_lite"])
    ap.add_argument("--threshold", type=float, default=4.0,
                    help="Score threshold to write a result to JSONL.")
    ap.add_argument("--must-have", action="append",
                    help="If set, only write results that contain this substring (repeatable).")
    ap.add_argument("--sample-len", type=int, default=280)
    ap.add_argument("--report-every", type=int, default=200,
                    help="Every N matrices per worker, append a progress JSON object. Set 0 to disable.")
    args = ap.parse_args()

    sources = parse_sources(args)
    if not sources:
        sources = [Source(name=f"src{i}", raw=s) for i, s in enumerate(DEFAULT_SOURCES)]

    matrix_map = build_matrix_map(args.matrix_mode)
    matrix_variants = list(matrix_map.keys())

    out_path = args.out
    lock_path = args.lock.strip() if args.lock and args.lock.strip() else None

    header = {
        "ts": time.time(),
        "type": "header",
        "hardcoded_params": HARD_CFG,
        "sources": [{"name": s.name, "raw_len": len(s.raw)} for s in sources],
        "matrix_mode": args.matrix_mode,
        "matrix_variants": len(matrix_variants),
        "workers": args.workers,
        "threshold": args.threshold,
        "must_have": args.must_have or [],
        "sample_len": args.sample_len,
    }
    append_jsonl(out_path, header, lock_path=lock_path)

    buckets = split_evenly(matrix_variants, args.workers)

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        signal.signal(signal.SIGINT, original_sigint)
        futs = []
        sources_payload = [(s.name, s.raw) for s in sources]
        for wid, bucket in enumerate(buckets):
            futs.append(ex.submit(
                worker_run,
                wid,
                bucket,
                sources_payload,
                matrix_map,
                out_path,
                lock_path,
                args.threshold,
                args.must_have,
                args.sample_len,
                args.report_every,
            ))
        try:
            for fut in as_completed(futs):
                results.append(fut.result())
        except KeyboardInterrupt:
            pass

    t1 = time.time()

    all_top: List[Tuple[float, Dict[str, Any]]] = []
    for r in results:
        for rec in r.get("top", []):
            all_top.append((rec.get("score_combo", -1e9), rec))
    all_top.sort(key=lambda x: x[0], reverse=True)
    all_top = all_top[:80]

    footer = {
        "ts": time.time(),
        "type": "footer",
        "elapsed_sec": round(t1 - t0, 3),
        "workers_done": len(results),
        "top_overall": [rec for _, rec in all_top],
    }
    append_jsonl(out_path, footer, lock_path=lock_path)

    print(f"[done] wrote: {out_path}")
    print(f"[done] elapsed: {t1 - t0:.2f}s, matrices: {len(matrix_variants)}")

if __name__ == "__main__":
    main()
