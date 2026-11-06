# Score every method's outputs on the same report ids with RadEval + finding-level metrics.

# Usage:
#   python evaluate.py --dataset iu_xray --split_file splits/iu_xray_split.json
#   python evaluate.py --dataset chexpert --split_file splits/chexpert_split.json --results results_chexpert/
#   python evaluate.py ... --add "greedy+radflag=results/greedy_radflag_p1_test.jsonl"

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="iu_xray", choices=["iu_xray", "chexpert"])
ap.add_argument("--split_file", required=True)
ap.add_argument("--split_name", default="test", choices=["dev", "cal", "test"])
ap.add_argument("--results", default="results/", help="folder with the generated / edited reports")
ap.add_argument("--only", default=None, help="comma-separated method names to score (default: all found)")
ap.add_argument("--add", action="append", default=[], help='extra method: "name=path1.jsonl,path2.jsonl" (repeatable)')
ap.add_argument("--no_clean", action="store_true")
ap.add_argument("--no_bertscore", action="store_true", help="skip BERTScore (slowest metric)")
ap.add_argument("--out_prefix", default=None)
args = ap.parse_args()

# default: each decoder with and without SecondLook; missing files are skipped
R, S = args.results, args.split_name
METHODS = {}
for m in ("greedy", "vcd", "dola", "opera", "ccd"):
    METHODS[m] = [os.path.join(R, f"{m}_findings_p1_{S}.jsonl")]
    METHODS[m + "+secondlook"] = [os.path.join(R, f"{m}_v5_p1_{S}.jsonl")]
for spec in args.add:
    name, paths = spec.split("=", 1)
    METHODS[name.strip()] = [p.strip() for p in paths.split(",")]
if args.only:
    keep = {m.strip() for m in args.only.split(",")}
    METHODS = {k: v for k, v in METHODS.items() if k in keep}

# split ids
if args.dataset == "chexpert":
    from chexpert_data import get_chexpert
    ds = get_chexpert()
else:
    from iu_xray_data import get_dataset
    ds = get_dataset("iu_xray")
split_ids = {str(ds[i].get("id", i)) for i in json.load(open(args.split_file))[args.split_name]}
del ds

# load outputs
outputs = {}
for name, paths in METHODS.items():
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[skip] {name}: missing {missing}")
        continue
    recs = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r["id"] in split_ids:
                    recs[r["id"]] = r
    outputs[name] = recs
    print(f"[load] {name}: {len(recs)} reports in split '{args.split_name}'")

if not outputs:
    sys.exit("No method outputs found.")
common = sorted(set.intersection(*(set(v) for v in outputs.values())))
print(f"\nPaired comparison on {len(common)} ids common to all {len(outputs)} methods (split has {len(split_ids)}).")
for name, recs in outputs.items():
    if len(recs) != len(common):
        print(f"  note: {name} has {len(recs) - len(common)} extra ids not scored")


def clean(t):
    if args.no_clean:
        return t.strip()
    t = re.sub(r"^\s*okay[^\n]*\n", "", t, flags=re.I)  # chatty opener line
    t = re.sub(r"[*#`_]+", " ", t)                        # markdown
    return re.sub(r"\s+", " ", t).strip() or "."          # empty -> "." so metrics don't crash


def build_evaluator():
    # RadEval v1 API, falling back to v2
    try:
        from RadEval import RadEval
        ev = RadEval(do_radgraph=True, do_bleu=True, do_rouge=True,
                     do_bertscore=not args.no_bertscore, do_chexbert=True, do_details=True)
        return ev, "v1"
    except (ImportError, TypeError) as e:
        print(f"[radeval] v1 API unavailable ({type(e).__name__}: {e}); trying v2")
    from radeval import RadEval
    metrics = ["radgraph", "bleu", "rouge", "f1chexbert"] + ([] if args.no_bertscore else ["bertscore"])
    return RadEval(metrics=metrics, detailed=True), "v2"


evaluator, api = build_evaluator()
print(f"[radeval] using {api} API")

# newer scikit-learn returns 3 values from _check_targets; RadEval expects 4
try:
    import RadEval.factual.f1chexbert as _fcb
    _orig_check_targets = _fcb._check_targets

    def _check_targets_compat(*a, **k):
        out = _orig_check_targets(*a, **k)
        return out if len(out) == 4 else (*out, None)

    _fcb._check_targets = _check_targets_compat
except (ImportError, AttributeError):
    pass

from finding_metrics import annotation_list_to_findings, compute_finding_level_metrics  # noqa: E402

# keywords of the verifier's finding types, for the false-positive breakdown
VERIFIABLE = ["effusion", "pneumothora", "opacit", "consolidat", "airspace", "infiltrat", "pneumonia",
              "atelecta", "edema", "interstitial", "reticular", "congestion", "engorgement", "cardiomegaly",
              "nodul", "mass", "fractur", "granulom", "degenerat", "spondyl", "osteophyt"]


def fp_breakdown(raw):
    # false-positive terms, and how many fall in the verifier's finding types
    hyp = annotation_list_to_findings(raw["radgraph"]["hypothesis_annotation_lists"])
    ref = annotation_list_to_findings(raw["radgraph"]["reference_annotation_lists"])
    fps = Counter()
    for g, r in zip(hyp, ref):
        fps.update({e for e, a in g if a == "present"} - {e for e, a in r if a == "present"})
    total = sum(fps.values())
    verif = sum(c for t, c in fps.items() if any(k in t for k in VERIFIABLE))
    return fps, total, verif


def flatten(d, prefix=""):
    # keep scalar metrics only
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(flatten(v, key + "/"))
    return out


# scores
all_scores, fp_tables, per_sample = {}, {}, {}
for name, recs in outputs.items():
    hyps = [clean(recs[i]["prediction"]) for i in common]
    refs = [clean(recs[i]["ground_truth"]) for i in common]
    print(f"\n=== scoring {name} ({len(hyps)} reports) ===")
    raw = evaluator(refs=refs, hyps=hyps)
    scores = flatten(raw)
    try:
        fl = compute_finding_level_metrics(raw, modes=("head_only",))["head_only"]
        per_sample[name] = [[s["tp"], s["fp"], s["fn"]] for s in fl["per_sample"]]
        for k in ("precision", "recall", "f1", "hallucination_rate", "total_tp", "total_fp", "total_fn"):
            scores[f"finding/{k}"] = float(fl[k])
        fps, fp_total, fp_verif = fp_breakdown(raw)
        scores["finding/fp_verifiable_share"] = fp_verif / fp_total if fp_total else float("nan")
        fp_tables[name] = fps.most_common(40)
        print(f"[fp] {name}: {fp_total} false positives, {fp_verif} ({100 * fp_verif / max(fp_total, 1):.1f}%) "
              f"in the verifier's finding types. Top 15: {fps.most_common(15)}")
    except (KeyError, TypeError) as e:
        print(f"[warn] finding-level metrics skipped for {name}: needs RadEval v1 detailed RadGraph output ({e})")
    all_scores[name] = scores
    print(json.dumps({k: round(v, 4) for k, v in scores.items()}, indent=1)[:2000])

# save
prefix = args.out_prefix or os.path.join(R, f"scores_{args.split_name}")
with open(prefix + ".json", "w") as f:
    json.dump({"dataset": args.dataset, "split": args.split_name, "n_ids": len(common), "ids": common,
               "cleaned": not args.no_clean, "radeval_api": api, "scores": all_scores,
               "top_false_positive_terms": fp_tables}, f, indent=1)
with open(prefix + "_per_sample.json", "w") as f:  # input for significance.py
    json.dump({"ids": common, "columns": ["tp", "fp", "fn"], "per_sample": per_sample}, f)
keys = sorted({k for s in all_scores.values() for k in s})
with open(prefix + ".csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method"] + keys)
    for name, s in all_scores.items():
        w.writerow([name] + [round(s.get(k, float("nan")), 4) for k in keys])


def pick(s, *needles):
    for k in s:
        if all(n in k.lower() for n in needles):
            return s[k]
    return float("nan")


nan = float("nan")
print("\n" + "=" * 100)
print(f"{'method':24} {'BLEU-4':>7} {'ROUGE-L':>8} {'BERTSc':>7} {'RG-part':>8} {'CXB5-mi':>8} {'CXB14-mi':>9} "
      f"{'find-P':>7} {'find-R':>7} {'find-F1':>8} {'H%':>7} {'FP-verif%':>10}")
for name, s in all_scores.items():
    print(f"{name:24} {pick(s, 'bleu_4'):7.4f} {pick(s, 'rougel'):8.4f} {pick(s, 'bertscore'):7.4f} "
          f"{pick(s, 'radgraph', 'partial'):8.4f} {pick(s, 'chexbert-5_micro'):8.4f} "
          f"{pick(s, 'chexbert-all_micro'):9.4f} {s.get('finding/precision', nan):7.4f} "
          f"{s.get('finding/recall', nan):7.4f} {s.get('finding/f1', nan):8.4f} "
          f"{100 * s.get('finding/hallucination_rate', nan):6.1f}% "
          f"{100 * s.get('finding/fp_verifiable_share', nan):9.1f}%")
print(f"\nfull metrics: {prefix}.csv / {prefix}.json")
print("CXB5-mi / CXB14-mi = CheXbert micro-F1 over 5 / 14 labels; H = unsupported-finding rate (1 - find-P); "
      "FP-verif% = share of false positives in the verifier's finding types.")