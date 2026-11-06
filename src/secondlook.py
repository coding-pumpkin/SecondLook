# SecondLook: combine track record, re-check and uncertainty per claim, then delete unsupported claims.

#   python secondlook_v5.py --cal_rows results/verify_greedy_cal_rows.jsonl \
#       --test results/greedy_verified_p1_test.jsonl,results/vcd_verified_p1_test.jsonl


# --cal_rows  verify.py calibrate output on the cal split (labelled claims)
# --test      verify.py apply outputs (signals in "verification"), or plain generation files (track record only)
# --features  prior,yesno,entropy (default) or a subset for the ablation

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

from claims import affirmed_groups, edit_report

ap = argparse.ArgumentParser()
ap.add_argument("--cal_rows", required=True)
ap.add_argument("--test", required=True, help="comma-separated")
ap.add_argument("--features", default="prior,yesno,entropy")
ap.add_argument("--suffix", default="", help="added to the output name, e.g. _prioronly")
ap.add_argument("--backbone", default=os.environ.get("BACKBONE", "medgemma"))
ap.add_argument("--threshold", type=float, default=None, help="analysis only: override the cal-chosen threshold")
args = ap.parse_args()
FEATS = [f for f in args.features.split(",") if f]
assert set(FEATS) <= {"prior", "yesno", "entropy"} and FEATS, "features: prior,yesno,entropy"


def fit_logreg(X, y, l2=1.0, iters=600, lr=0.1):
    # plain gradient descent, L2 on the weights (not the bias); returns [w..., bias]
    d = len(X[0])
    w = [0.0] * (d + 1)
    for _ in range(iters):
        g = [0.0] * (d + 1)
        for xi, yi in zip(X, y):
            z = w[-1] + sum(a * b for a, b in zip(w, xi))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            for j in range(d):
                g[j] += (p - yi) * xi[j]
            g[-1] += p - yi
        for j in range(d):
            w[j] -= lr * (g[j] / len(X) + l2 * w[j] / len(X))
        w[-1] -= lr * g[-1] / len(X)
    return w


def build(fit):
    # track record per type, standardization and logistic model, all from the `fit` claims
    by = defaultdict(lambda: [0, 0])
    for c in fit:
        by[c["group"]][0] += c["y"]
        by[c["group"]][1] += 1
    base = sum(c["y"] for c in fit) / len(fit)
    prior = lambda g: (by[g][0] + 2 * base) / (by[g][1] + 2)  # two pseudo-claims at the overall rate

    def stats(key):
        v = [c[key] for c in fit if c.get(key) is not None]
        m = sum(v) / len(v) if v else 0.0
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 if len(v) > 1 else 1.0
        return m, sd or 1.0
    sc = {k: stats(k) for k in ("yesno", "entropy")}

    def x(c):
        out = []
        if "prior" in FEATS:
            p = prior(c["group"])
            out.append(math.log(p / (1 - p)))
        for k in ("yesno", "entropy"):
            if k in FEATS:
                v = c.get(k)
                out += [0.0 if v is None else (v - sc[k][0]) / sc[k][1], float(v is None)]  # value, missing flag
        return out
    w = fit_logreg([x(c) for c in fit], [c["y"] for c in fit])
    prob = lambda c: 1 / (1 + math.exp(-max(-30, min(30, w[-1] + sum(a * b for a, b in zip(w, x(c)))))))
    return prob, w, {g: (by[g][0], by[g][1], prior(g)) for g in by}


def claim_f1(pairs, t):
    # pairs: (prob, supported); keep claims with prob >= t
    kept = [y for p, y in pairs if p >= t]
    tp, npos = sum(kept), sum(y for _, y in pairs)
    prec = tp / len(kept) if kept else 0.0
    rec = tp / npos if npos else 0.0
    return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), prec, rec, 1 - len(kept) / len(pairs)


# Fit on cal
cal = [r for r in map(json.loads, open(args.cal_rows)) if r.get("group") and r.get("ref_status") in ("pos", "neg")]
C = [{"id": r["id"], "group": r["group"], "y": int(r["ref_status"] == "pos"),
      "yesno": r.get("score"), "entropy": r.get("entropy")} for r in cal]
print(f"cal: {len(C)} labelled claims, {sum(c['y'] for c in C)} supported "
      f"(precision {100 * sum(c['y'] for c in C) / len(C):.1f}%); features: {FEATS}")

# threshold from cross-fitted predictions (report-level 2-fold x 10), so it is not tuned on its own claims
ids = sorted({c["id"] for c in C})
oof = defaultdict(list)
for rep in range(10):
    sh = ids[:]
    random.Random(rep).shuffle(sh)
    half = set(sh[: len(sh) // 2])
    for a in (True, False):
        prob, _, _ = build([c for c in C if (c["id"] in half) == a])
        for i, c in enumerate(C):
            if (c["id"] in half) != a:
                oof[i].append(prob(c))
pairs = [(sum(oof[i]) / len(oof[i]), c["y"]) for i, c in enumerate(C)]
grid = sorted({round(p, 3) for p, _ in pairs})
(f1, prec, rec, drop), T = max(((claim_f1(pairs, t), t) for t in [0.0] + grid), key=lambda z: z[0][0])
f1_0, prec_0, _, _ = claim_f1(pairs, 0.0)
print(f"threshold: drop if p(supported) < {T:.3f}")
print(f"  cal (out-of-sample): {100 * drop:.0f}% of claims dropped; precision {100 * prec_0:.1f} -> {100 * prec:.1f}; "
      f"supported kept {100 * rec:.1f}%; claim F1 {100 * f1_0:.1f} -> {100 * f1:.1f}")
if args.threshold is not None:
    print(f"ANALYSIS RUN: threshold {T:.3f} -> {args.threshold:.3f}")
    T = args.threshold

prob, w, table = build(C)  # final model on all of cal
names = (["prior(logit)"] if "prior" in FEATS else []) + \
        sum([[k, k + "_missing"] for k in ("yesno", "entropy") if k in FEATS], [])
print("weights: " + ", ".join(f"{n}={v:+.2f}" for n, v in zip(names, w)) + f", bias={w[-1]:+.2f}")
print("track record on cal (supported / claimed):")
for g, (s, n, p) in sorted(table.items(), key=lambda z: z[1][2]):
    print(f"  {g:13s} {s:4d}/{n:<4d} = {100 * s / n:5.1f}%   (smoothed {100 * p:.1f}%)")

#Apply to test
for path in [p for p in args.test.split(",") if p]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    method = os.path.basename(path).split("_")[0]
    out = os.path.join(os.path.dirname(path) or ".", f"{method}_v5{args.suffix}_p1_test.jsonl")
    acts, per_type, changed = Counter(), defaultdict(Counter), 0
    with open(out, "w") as f:
        for r in rows:
            text = r.get("original_prediction", r["prediction"])
            ver = {v["group"]: v for v in r.get("verification", [])}
            decisions, info = {}, []
            for g in affirmed_groups(text):
                v = ver.get(g, {})
                c = {"group": g, "yesno": v.get("score"), "entropy": v.get("entropy")}
                p = prob(c)
                a = "drop" if p < T else "keep"
                decisions[g] = a
                acts[a] += 1
                per_type[g][a] += 1
                info.append({"group": g, "p_supported": round(p, 4), "yesno": c["yesno"],
                             "entropy": c["entropy"], "action": a})
            edited = edit_report(text, decisions) if decisions else text
            changed += edited != text
            f.write(json.dumps({"id": r["id"], "backbone": r.get("backbone", args.backbone),
                                "mode": f"{method}+v5{args.suffix}", "prompt": r.get("prompt", ""),
                                "prediction": edited, "original_prediction": text,
                                "ground_truth": r["ground_truth"], "v5": info}) + "\n")
    print(f"\n{path}: {len(rows)} reports, {changed} edited; claims {dict(acts)}")
    print("  dropped by type: " + ", ".join(f"{g} {per_type[g]['drop']}/{sum(per_type[g].values())}"
                                            for g in sorted(per_type, key=lambda g: -per_type[g]['drop'])
                                            if per_type[g]["drop"]))
    print(f"  -> {out}")