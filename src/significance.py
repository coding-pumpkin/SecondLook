# Paired bootstrap significance for "X" vs "X+secondlook" on the finding-level metrics.

import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--per_sample", default="results/scores_test_per_sample.json")
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--base", default=None,
                help="compare every other system against this one (e.g. greedy for the ablation table); "
                     "default: pair X with X+secondlook")
ap.add_argument("--pairs", default=None, help='explicit pairs, e.g. "greedy:greedy+secondlook,opera:opera+secondlook"')
args = ap.parse_args()

d = json.load(open(args.per_sample))
ps = d["per_sample"]
n_rep = len(d["ids"])


def metrics(rows, idx):
    tp = sum(rows[i][0] for i in idx)
    fp = sum(rows[i][1] for i in idx)
    fn = sum(rows[i][2] for i in idx)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"halluc": fp / (tp + fp) if tp + fp else 0.0, "precision": p, "recall": r, "f1": f1}


rng = random.Random(args.seed)
boot = [[rng.randrange(n_rep) for _ in range(n_rep)] for _ in range(args.n)]
full = list(range(n_rep))

if args.pairs:
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
elif args.base:
    pairs = [(args.base, m) for m in ps if m != args.base]
else:
    pairs = [(m, m + "+secondlook") for m in ps if not m.endswith("+secondlook") and m + "+secondlook" in ps]

print(f"{n_rep} reports, {args.n} paired bootstrap resamples\n")
print(f"{'comparison':34} {'metric':10} {'base':>8} {'other':>8} {'diff':>8}   {'95% CI of diff':>20}  {'p':>7}")
for base, ver in pairs:
    a, b = ps[base], ps[ver]
    for metric in ("halluc", "precision", "recall", "f1"):
        m0, m1 = metrics(a, full)[metric], metrics(b, full)[metric]
        diffs = sorted(metrics(b, idx)[metric] - metrics(a, idx)[metric] for idx in boot)
        lo, hi = diffs[int(0.025 * args.n)], diffs[int(0.975 * args.n)]
        # two-sided p: how often the resampled difference crosses zero
        p = 2 * min(sum(x <= 0 for x in diffs), sum(x >= 0 for x in diffs)) / args.n
        print(f"{base + ' vs ' + ver:34} {metric:10} {m0:8.4f} {m1:8.4f} {m1 - m0:+8.4f}   "
              f"[{lo:+.4f}, {hi:+.4f}]  {min(p, 1.0):7.4f}")
    print()