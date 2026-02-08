from __future__ import annotations

import re, string
from typing import List, Tuple

# jekhr_letters = [
#     "^",
#     "V",
#     "<",
#     ")",
#     "-",
#     "=",
#     "8",
#     "T",
#     "I",
#     "J",
#     "X",
#     "(",
#     "W",
#     '"',
#     "O",
#     "]",
#     "Q",
#     "B",
#     "~",
#     "L",
#     "C",
#     ">",
#     "U",
#     "+",
#     "\\",
#     "/",
# ]
AZ = string.ascii_uppercase
# AZ = "".join(jekhr_letters)

BASE_MATRIX_4x4 = [
    [3, 1, 6, 1],
    [1, 2, 1, 1],
    [1, 1, 3, 1],
    [4, 6, 1, 1],
]

MATRIX_PERMUTATED = [[1, 6, 4, 1], [1, 2, 1, 1], [3, 1, 1, 1], [6, 1, 3, 1]]

# BASE_MATRIX_4x4 = [[1, 1, 1, 1], [1, 3, 6, 1], [1, 1, 4, 1], [4, 6, 1, 1]]

BOOK2_ROWS = [
    ["ljkhbl", "nilse", "jfpce", "ojvco", "ld"],
    ["slcld", "ylddiv", "dnolsd", "dd", "sd"],
    ["sdcp", "cppcs", "cccpc", "cpsc"],
    ["awdp", "cpcw", "cfw", "ce"],
    ["cpvc", "ev", "vcemmev", "vrvf"],
    ["cp", "fd", "vmfpm", "xcv"],
]
BOOK2_FLAT = [b for row in BOOK2_ROWS for b in row]


def map_to_AZ(idx: int) -> str:
    return AZ[idx % 26]


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


def normalize_digits_keep_boundaries(s: str) -> Tuple[str, List[int]]:
    digits = []
    boundaries = []
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


def chunk_by_fixed_width(
    digits: str, width: int, offset: int, boundaries: List[int] | None
) -> List[str]:
    out = []
    i = offset
    L = len(digits)
    bset = set(boundaries or [])
    while i + width <= L:
        if boundaries:
            internal = [b for b in bset if i < b < i + width]
            if internal:
                i = min(internal)
                continue
        out.append(digits[i : i + width])
        i += width
    return out

def row_index_from_v(v: int, row_map: str) -> int:
    if row_map == "v-1":
        return v - 1
    elif row_map == "v%6":
        return v % 6
    else:
        return (v - 1) % 6
    
def decode_matrix_book2(
    raw: str,
    *,
    token_w: int,
    offset: int,
    use_boundaries: bool,
    coord_mode: str,
    row_map: str,
    col_rule: str,
) -> str:
    digits, bounds = normalize_digits_keep_boundaries(raw)
    mat = MATRIX_PERMUTATED

    toks = chunk_by_fixed_width(
        digits, token_w, offset, bounds if use_boundaries else None
    )
    out: List[str] = []

    for t in toks:
        if len(t) < 4:
            continue
        
        a, b, c, d = (ord(t[0]) - 48, ord(t[1]) - 48, ord(t[2]) - 48, ord(t[3]) - 48)

        # "11" = 1+1, i think it's a sign to use mathemagics
        # d % 4 chooses the operation to perform, in this case, the mirroring 
        if t.startswith("11") and (d % 4 == 1):
            r = (a - b) % 4
            cc = (c - d) % 4
            v = mat[r][cc]

            row_i = row_index_from_v(v, row_map)
            row = BOOK2_ROWS[row_i]
            L = len(row)
            col = c % L

            block = row[col]
            idx = BOOK2_FLAT.index(block)
            out.append(map_to_AZ(idx))

            # 94 = mirror: swap this output with the previous output
            if len(out) >= 2:
                out[-2], out[-1] = out[-1], out[-2]
            continue

        if coord_mode == "ab":
            r, cc = a % 4, b % 4
        else:
            r, cc = b % 4, a % 4

        v = mat[r][cc]
        row_i = row_index_from_v(v, row_map)
        row = BOOK2_ROWS[row_i]
        print(
            f"Token: {t} -> a={a} b={b} c={c} d={d} - {v} - {row_i} - {row} - {c} - {d}"
        )
        L = len(row)
        if L <= 0:
            continue

        if col_rule == "c-d":
            col = (c - d) % L
        elif col_rule == "d-c":
            if c == d:
                col = (10 * c + d) % L  # "49-ish": treat as a 2-digit number. Basically undo the splitting of "49" into "4" and "9" if they happen to be the same digit and handle as pure decimal 49 (operation 1+1 = 49)
            else:
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
            idx = sum(ord(x) for x in block) % 26

        out.append(map_to_AZ(idx))

    return "".join(out)


def main():
    sources = [
        "6468895219911800651288952364672119118",
    ]

    sources = [(f"src{i}", s) for i, s in enumerate(sources)]
    for name, raw in sources:
        out = decode_matrix_book2(
            raw,
            token_w=4,
            offset=3,
            use_boundaries=True,
            coord_mode="ab",
            row_map="v-1",
            col_rule="d-c",
        )
        print(f"\n=== {name} ===")
        print(raw)
        print(out)


if __name__ == "__main__":
    main()
