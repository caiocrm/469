just run python solve.py and the grid.txt will be generated.

the community can use this script to generate the grid and try after reordering, or even making books overlapping.


# Code to check robustness of the decoding
```

from pathlib import Path
import numpy as np
import zlib
from numpy.linalg import svd

def load_cols_as_binary(path, ncols=3):
    cols = [[] for _ in range(ncols)]
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("-"):
            continue
        parts = line.split("|")
        if len(parts) < ncols:
            continue
        for i in range(ncols):
            cols[i].append(parts[i])

    widths = [max(len(s) for s in c) for c in cols]
    mats = []
    for c, w in zip(cols, widths):
        M = np.zeros((len(c), w), dtype=np.uint8)
        for r, s in enumerate(c):
            s = s.ljust(w)
            M[r, :] = [0 if ch == " " else 1 for ch in s[:w]]
        mats.append(M)
    return mats

def shuffle_within_rows(M, rng):
    Msh = M.copy()
    for r in range(Msh.shape[0]):
        rng.shuffle(Msh[r])
    return Msh

def stat_top_singular(M):
    X = M.astype(float)
    X -= X.mean()
    return svd(X, compute_uv=False)[0]

def stat_compress_ratio(M):
    b = bytes(M.flatten().tolist())
    return len(zlib.compress(b, level=9)) / len(b)

def perm_test(M, stat_fn, n=5000, seed=0, side="greater"):
    rng = np.random.default_rng(seed)
    obs = stat_fn(M)
    vals = np.array([stat_fn(shuffle_within_rows(M, rng)) for _ in range(n)])
    if side == "greater":
        p = (np.sum(vals >= obs) + 1) / (n + 1)
    else:
        p = (np.sum(vals <= obs) + 1) / (n + 1)
    z = (obs - vals.mean()) / vals.std(ddof=1)
    return obs, vals.mean(), vals.std(ddof=1), z, p

# --- run ---
M_3478, M_ones_real, M_ones_random = load_cols_as_binary("data/grid_and_random_after_2column.txt")

for name, M in [
    ("3478", M_3478),
    ("ones_real", M_ones_real),
    ("ones_random", M_ones_random),
]:
    obs, mu, sd, z, p = perm_test(M, stat_top_singular, n=5000, seed=1, side="greater")
    print(f"[{name}] topSV obs={obs:.3f} null={mu:.3f}±{sd:.3f} z={z:.2f} p={p:.6f}")

    obs, mu, sd, z, p = perm_test(M, stat_compress_ratio, n=5000, seed=2, side="less")
    print(f"[{name}] compress obs={obs:.4f} null={mu:.4f}±{sd:.4f} z={z:.2f} p={p:.6f}")
    print()


```

Results:
```
[3478] topSV obs=12.183 null=11.343±0.139 z=6.05 p=0.000200
[3478] compress obs=0.0771 null=0.1576±0.0008 z=-99.49 p=0.000200

[ones_real] topSV obs=8.903 null=8.509±0.144 z=2.75 p=0.007798
[ones_real] compress obs=0.0824 null=0.1103±0.0007 z=-39.15 p=0.000200

[ones_random] topSV obs=7.090 null=7.031±0.123 z=0.48 p=0.296541
[ones_random] compress obs=0.0844 null=0.0847±0.0007 z=-0.46 p=0.348330
```

# Chat GPT interpretation of results:
Verdict: **your “pattern” columns are *very* unlikely to be explainable by chance under the matched-shuffle null — and your random-control column behaves exactly like noise.** That’s the clean outcome you want.

Here’s how to read your numbers.

## 1) The key comparison

You’re testing against a strong null: **same rows, same per-row pixel counts, but scrambled within each row**. So any “it’s just because there are lots of digits” objection is mostly neutralized.

### `3478` column

* **topSV p = 0.000200, z = 6.05**
* **compress p = 0.000200, z = -99.49**

That’s *extremely* significant.
Interpretation:

* **SVD**: there’s unusually strong global alignment/low-rank structure (image-like structure).
* **Compression**: it’s massively more regular than a shuffled version with the same per-row density.

### `ones_real` column

* **topSV p ≈ 0.0078, z = 2.75**
* **compress p = 0.000200, z = -39.15**

Also significant, though weaker than `3478` on the SVD measure. Still: the compression result is screaming “non-random structure.”

### `ones_random` column (your control)

* **topSV p ≈ 0.297**
* **compress p ≈ 0.348**

This is what “no pattern” looks like: **not significant, consistent with the null**.

So: **your method distinguishes “real signal” vs “random noise” correctly.**

## 2) What “p = 0.000200” really means here

With permutation tests, p-values are discrete. If you ran **5000** shuffles, the smallest possible p is about **1/(5000+1) ≈ 0.0002**.
So `0.000200` means: **none of your 5000 shuffled trials beat the real data** (or at most 0, depending on the +1 correction).

That’s not “kinda significant.” That’s “your observed structure is beyond the tail of the null distribution.”

## 3) Why the compression z-scores are so huge

Those z’s are *effect size in standard deviations*, and the null standard deviation is tiny (because compression ratio varies very little under shuffling at fixed density). So even moderate absolute differences become enormous z-scores.

If you want something more intuitive to show people, report an effect size like:

* **Δcompression = null_mean − observed**
* **% improvement = (null_mean − observed) / null_mean**

For `3478`:
`0.1576 − 0.0771 = 0.0805` → about **51% lower compression ratio** than null (much more compressible).

## 4) What you can honestly claim

You can say:

* **Under a matched randomization test**, the `3478` projection is **inconsistent with chance** (p ≤ 0.0002) by two independent measures (low-rank structure and compressibility).
* Your **random-generated control** behaves as expected (p ~ 0.3), suggesting the test isn’t just hallucinating patterns everywhere.

What you *cannot* claim from this alone:

* “It’s specifically a beholder.”
  This proves “structured signal,” not “this exact semantic image.” Semantic identification needs template matching / human-blind tests / cross-book replication.

## 5) Multiple perspectives: how skeptics might push back (and how to answer)

### “You cherry-picked 3478.”

Answer: run an **automatic search over digit subsets** and show `3478` is near-optimal *with multiple-testing correction*. Or pre-register: “Here are the subsets I’m testing.”

### “Your null model is wrong.”

Your null is already strong (preserves per-row counts). If you want to be even more conservative, use a null that preserves **per-row AND per-column counts** (a bipartite swap / “matrix degree-preserving randomization”). Harder to implement, but even more bulletproof.

### “Compression can be gamed.”

That’s why your **SVD test** agreeing is important: two different statistics, same conclusion.

## Novel ideas to make it land even harder

* **Holdout test:** compute a “pattern template” from half the books and show it predicts the other half better than shuffled nulls. That shuts down “overfit/cherry-pick” arguments.
* **Degree-preserving null:** randomize the binary matrix while preserving row sums *and* column sums (swap randomization). If `3478` still wins, it’s ironclad.
* **Blind judging:** mix real vs shuffled vs random images; ask people to pick which looks “structured.” If they succeed at high accuracy, it backs up the stats with human perception.

If you want, paste your shuffle count (`n=`) and I’ll translate the p-values into plain-language statements + a short “methods paragraph” you can post so people stop arguing about vibes.
