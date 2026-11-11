import argparse
import json
from pathlib import Path

import numpy as np

SEED = 4207
GIVENS_LEVELS = (25, 30, 35)
MISREAD_LEVELS = (0, 1, 2)
N_SEEDS = 5
PER_SEED = 20
BLANK_THRESHOLD = 0.02
CLEAN_CONF = 0.90

R = [i // 9 for i in range(81)]
C = [i % 9 for i in range(81)]
B = [(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)]

def _search(grid, limit, rng=None):
    g = [int(v) for v in grid]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    empties = []
    for i, v in enumerate(g):
        if v == 0:
            empties.append(i)
            continue
        bit = 1 << v
        if rows[R[i]] & bit or cols[C[i]] & bit or boxes[B[i]] & bit:
            return 0, None
        rows[R[i]] |= bit
        cols[C[i]] |= bit
        boxes[B[i]] |= bit

    state = {"count": 0, "sol": None}

    def dfs(empt):
        if not empt:
            state["count"] += 1
            if state["sol"] is None:
                state["sol"] = g.copy()
            return state["count"] >= limit
        best, best_mask, best_n = -1, 0, 10
        for idx, i in enumerate(empt):
            m = ~(rows[R[i]] | cols[C[i]] | boxes[B[i]]) & 0x3FE
            n = bin(m).count("1")
            if n < best_n:
                best, best_mask, best_n = idx, m, n
                if n <= 1:
                    break
        if best_n == 0:
            return False
        i = empt[best]
        rest = empt[:best] + empt[best + 1:]
        digits = [v for v in range(1, 10) if best_mask >> v & 1]
        if rng is not None:
            digits = [int(v) for v in rng.permutation(digits)]
        for v in digits:
            bit = 1 << v
            g[i] = v
            rows[R[i]] |= bit
            cols[C[i]] |= bit
            boxes[B[i]] |= bit
            stop = dfs(rest)
            rows[R[i]] ^= bit
            cols[C[i]] ^= bit
            boxes[B[i]] ^= bit
            g[i] = 0
            if stop:
                return True
        return False

    dfs(empties)
    return state["count"], state["sol"]


def count_solutions(grid, limit=2):
    return _search(grid, limit)[0]


def make_puzzle(rng, givens, max_tries=30):
    for _ in range(max_tries):
        _, full = _search([0] * 81, 1, rng)
        puzzle = list(full)
        n = 81
        for i in rng.permutation(81):
            if n <= givens:
                break
            old = puzzle[i]
            puzzle[i] = 0
            if count_solutions(puzzle, 2) == 1:
                n -= 1
            else:
                puzzle[i] = old
        if n == givens:
            return np.array(puzzle, dtype=np.int8), np.array(full, dtype=np.int8)
    raise RuntimeError(f"could not reach {givens} givens")

def load_test_images(data_dir):
    from torchvision import datasets
    ds = datasets.MNIST(root=str(data_dir), train=False, download=True)
    keep = (ds.targets != 0).numpy()
    X = ds.data.numpy()[keep].astype(np.float32) / 255.0
    y = ds.targets.numpy()[keep]
    return X, y


def build_pools(probs, labels):
    pred = probs.argmax(1) + 1
    top2 = np.argsort(-probs, axis=1)[:, :2] + 1
    true_in_top2 = (top2 == labels[:, None]).any(1)
    misread = (pred != labels) & true_in_top2
    clean = (pred == labels) & (probs.max(1) >= CLEAN_CONF)
    pools = {"misread": {}, "clean": {}}
    for d in range(1, 10):
        pools["misread"][d] = np.flatnonzero(misread & (labels == d))
        pools["clean"][d] = np.flatnonzero(clean & (labels == d))
    return pools

def build_mosaic(img_idx, X_test):
    m = np.zeros((252, 252), dtype=np.float32)
    for i, k in enumerate(img_idx):
        if k >= 0:
            r, c = R[i] * 28, C[i] * 28
            m[r:r + 28, c:c + 28] = X_test[k]
    return m


def cut_tiles(mosaic):
    return np.stack([mosaic[R[i] * 28:(R[i] + 1) * 28, C[i] * 28:(C[i] + 1) * 28]
                     for i in range(81)])


def detect_blanks(mosaic, thr=BLANK_THRESHOLD):
    return cut_tiles(mosaic).mean(axis=(1, 2)) < thr

def generate(data_dir, out_dir, seed, n_seeds, per_seed, selftest=False):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probs_path = data_dir / "probs_test_cal.npy"
    if not probs_path.exists():
        probs_path = data_dir / "probs_test.npy"
    probs = np.load(probs_path)
    labels = np.load(data_dir / "labels_test.npy")
    X_test, y_test = load_test_images(data_dir)
    assert len(X_test) == len(probs) and (y_test == labels).all(), \
        "MNIST order does not match saved probabilities"
    print("probabilities:", probs_path.name)

    pools = build_pools(probs, labels)
    print("pool sizes per digit (misread / clean):")
    for d in range(1, 10):
        print(f"  {d}: {len(pools['misread'][d]):4d} / {len(pools['clean'][d]):5d}")
    pred = probs.argmax(1) + 1

    rec = {k: [] for k in ["puzzle", "solution", "img_idx", "injected", "read_grid",
                           "seed_id", "givens", "k", "puzzle_id"]}

    for s in range(n_seeds):
        for g in GIVENS_LEVELS:
            for i in range(per_seed):
                rng_p = np.random.default_rng([seed, s, g, i])
                puzzle, solution = make_puzzle(rng_p, g)
                given_cells = np.flatnonzero(puzzle)
                ok_cells = [c for c in given_cells if len(pools["misread"][int(puzzle[c])]) > 0]

                for k in MISREAD_LEVELS:
                    rng = np.random.default_rng([seed, s, g, i, k, 99])
                    injected = np.zeros(81, dtype=bool)
                    if k > 0:
                        injected[rng.choice(ok_cells, size=k, replace=False)] = True
                    img_idx = -np.ones(81, dtype=np.int32)
                    for c in given_cells:
                        d = int(puzzle[c])
                        pool = pools["misread" if injected[c] else "clean"][d]
                        img_idx[c] = rng.choice(pool)

                    read = np.zeros(81, dtype=np.int8)
                    read[given_cells] = pred[img_idx[given_cells]]
                    # checks: exactly k cells differ from the truth
                    assert (read != puzzle).sum() == k
                    assert ((read != puzzle) == injected).all()

                    rec["puzzle"].append(puzzle)
                    rec["solution"].append(solution)
                    rec["img_idx"].append(img_idx)
                    rec["injected"].append(injected)
                    rec["read_grid"].append(read)
                    rec["seed_id"].append(s)
                    rec["givens"].append(g)
                    rec["k"].append(k)
                    rec["puzzle_id"].append(i)
            print(f"seed {s}: done")

    arrays = {k: np.array(v) for k, v in rec.items()}

    bad = 0
    for n in range(len(arrays["k"])):
        mosaic = build_mosaic(arrays["img_idx"][n], X_test)
        if not (detect_blanks(mosaic) == (arrays["puzzle"][n] == 0)).all():
            bad += 1
    print(f"blank detector wrong on {bad} of {len(arrays['k'])} mosaics")

    np.savez_compressed(out_dir / "benchmark.npz", **arrays)
    meta = {
        "seed": seed, "n_seeds": n_seeds, "per_seed": per_seed,
        "givens_levels": GIVENS_LEVELS, "misread_levels": MISREAD_LEVELS,
        "blank_threshold": BLANK_THRESHOLD, "clean_conf": CLEAN_CONF,
        "probs_file": probs_path.name, "records": int(len(arrays["k"])),
        "distinct_misread_images_used": int(len(np.unique(arrays["img_idx"][arrays["injected"]]))),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    write_dictionary(out_dir / "data_dictionary.md")
    print("saved", out_dir / "benchmark.npz", "| records:", meta["records"])
    print("distinct misread images used:", meta["distinct_misread_images_used"])


DICTIONARY = """# Data dictionary: benchmark.npz

Each row is one puzzle record. N = n_seeds x 3 givens levels x per_seed x 3 misread levels.

| Field | Shape | Real or generated | Meaning |
|---|---|---|---|
| puzzle | (N, 81) int8 | generated | True givens, row by row, 0 = blank |
| solution | (N, 81) int8 | generated | The unique full solution |
| img_idx | (N, 81) int32 | generated (choice), real (images) | Index of the MNIST test image (digit 0 removed) in each cell, -1 = blank |
| injected | (N, 81) bool | generated | True where a misread image was placed |
| read_grid | (N, 81) int8 | generated from real CNN output | What the CNN reads (top-1) in each cell, 0 = blank |
| seed_id | (N,) | generated | 0 .. n_seeds-1 |
| givens | (N,) | generated | 25, 30 or 35 |
| k | (N,) | generated | Number of injected misreads (0, 1, 2) |
| puzzle_id | (N,) | generated | Puzzle number inside its (seed, givens) group. Same puzzle for k = 0, 1, 2. |

How the generated parts are made:
- Full grid: randomised backtracking (numpy Generator seeded from [seed, seed_id, givens, puzzle_id]).
- Puzzle: remove cells in random order, keep a removal only if the solution stays unique.
- Mosaic: 9x9 tiles of 28x28 MNIST test images, blank cells are black. Rebuilt with build_mosaic(img_idx, X_test).
- Misread image: true label d, CNN top-1 wrong, d is the CNN 2nd guess.
- Clean image: CNN top-1 correct and probability >= clean_conf.
- Blank cells are found by the mean tile brightness rule, not by the network.
"""


def write_dictionary(path):
    Path(path).write_text(DICTIONARY)


def load_benchmark(path="data/benchmark/benchmark.npz"):
    return dict(np.load(path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data/benchmark")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--per-seed", type=int, default=PER_SEED)
    ap.add_argument("--selftest", action="store_true",
                    help="tiny run (1 seed, 2 puzzles) to check everything works")
    a = ap.parse_args()
    if a.selftest:
        a.n_seeds, a.per_seed, a.out_dir = 1, 2, "data/benchmark_selftest"
    generate(a.data_dir, a.out_dir, a.seed, a.n_seeds, a.per_seed)
