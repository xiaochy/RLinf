"""Per-seed outcomes out of an RLinf eval's `eval_progress.jsonl`, split by evaluation pass.

Why this is not a one-liner over the file:

`eval_progress.jsonl` is appended to, and the trainer evaluates once per epoch, so one
directory's file can hold several complete or partial evaluation passes. Reading the whole
file into a dict keyed by seed silently mixes them -- a later pass overwrites an earlier
pass's outcome for every seed it reached. On 2026-11-12 that turned the SFT-100 baseline
from its real 0.760 into 0.840, because that job's process ran a second pass over 85 of the
100 seeds before its wall clock killed it.

A pass boundary is where `round` stops increasing. Rows carry their own `seeds`, so pairing
is by seed value and a truncated pass costs only the seeds it never reached.

    python parse_student_eval.py eval_logs_student_env/*/           # one line per pass
    python parse_student_eval.py --pairs eval_logs_*/*/             # and every comparison
"""

import argparse
import glob
import json
import os
from math import comb


def passes(path):
    """The file's evaluation passes, oldest first, as lists of rows."""
    out, cur, prev = [], [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rnd = row.get("round")
            if prev is not None and rnd is not None and rnd <= prev:
                out.append(cur)
                cur = []
            cur.append(row)
            prev = rnd
    if cur:
        out.append(cur)
    return out


def per_seed(rows, metric="success_once"):
    """{seed: 0 or 1} for one pass. Rows without the metric are a started-but-empty round."""
    got = {}
    for row in rows:
        if metric not in row or "seeds" not in row:
            continue
        seeds = [s for group in row["seeds"] for s in group]
        for seed, value in zip(seeds, row[metric]):
            got[seed] = int(value)
    return got


def mcnemar(a, b):
    common = sorted(set(a) & set(b))
    b01 = sum(1 for s in common if a[s] == 0 and b[s] == 1)
    b10 = sum(1 for s in common if a[s] == 1 and b[s] == 0)
    n = b01 + b10
    if n == 0:
        return len(common), 0, 0, 1.0
    k = min(b01, b10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return len(common), b01, b10, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="eval log directories, or their jsonl files")
    ap.add_argument("--metric", default="success_once")
    ap.add_argument("--pairs", action="store_true", help="every pairwise comparison")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    runs = []
    for item in sorted(sum((glob.glob(x) for x in args.dirs), [])):
        path = item if item.endswith(".jsonl") else os.path.join(item, "eval_progress.jsonl")
        if not os.path.exists(path):
            continue
        label = os.path.basename(os.path.dirname(path))
        found = passes(path)
        for i, rows in enumerate(found):
            got = per_seed(rows, args.metric)
            if not got:
                continue
            # Only the last pass of a file can have been cut short by the process dying;
            # an earlier one that is short was followed by a restart, which is the same thing.
            note = "" if len(got) == 100 else f"  PARTIAL -- {len(got)} seeds, not 100"
            tag = label if len(found) == 1 else f"{label} [pass {i + 1} of {len(found)}]"
            runs.append(dict(tag=tag, per_seed=got))
            print(f"{sum(got.values()):3d}/{len(got):3d} = {sum(got.values()) / len(got):.3f}  "
                  f"{tag}{note}")

    if args.pairs and len(runs) > 1:
        print()
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                x, y = runs[i], runs[j]
                n, b01, b10, p = mcnemar(x["per_seed"], y["per_seed"])
                xr = sum(x["per_seed"].values()) / len(x["per_seed"])
                yr = sum(y["per_seed"].values()) / len(y["per_seed"])
                print(f"common {n:3d}  +{b01:2d}/-{b10:2d}  d={yr - xr:+.3f}  p={p:.4f}   "
                      f"{x['tag']}  ->  {y['tag']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(runs, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
