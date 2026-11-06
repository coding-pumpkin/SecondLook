# SecondLook signals for every affirmed claim in a report:
#   re-check     mean P(yes) over yes/no questions asked about the image in a fresh context
#   uncertainty  mean normalized token entropy of the sentence that states the claim
#   calibrate  cal split; also labels each claim against the reference -> results/verify_<tag>_cal_rows.jsonl
#   apply      any split; stores the signals per report               -> results/<tag>_verified_p1_<split>.jsonl

'''  python verify.py --mode calibrate --preds results/greedy_findings_p1_cal.jsonl --tag greedy \\
       --split_file splits/iu_xray_split.json --split_name cal '''

import argparse
import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CheXagentAdapter, LlavaMedAdapter, MedGemmaAdapter  # noqa: E402
from claims import K, affirmed_groups, ref_support  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["calibrate", "apply"])
ap.add_argument("--preds", required=True, help="jsonl of reports (comma-separated for several files)")
ap.add_argument("--tag", required=True, help="name for outputs, e.g. greedy or vcd")
ap.add_argument("--dataset", default="iu_xray", choices=["iu_xray", "chexpert"])
ap.add_argument("--split_file", required=True)
ap.add_argument("--split_name", default="cal", choices=["dev", "cal", "test"])
ap.add_argument("--prompt_id", type=int, default=1, help="only used in the output file name")
ap.add_argument("--prior_max", type=float, default=0.1, help="bias filter: drop questions with P(yes) above this on a blank image")
ap.add_argument("--backbone", default=os.environ.get("BACKBONE", "medgemma"), choices=["medgemma", "chexagent", "llavamed"])
ap.add_argument("--model", default=None, help="checkpoint; default depends on --backbone")
ap.add_argument("--img_root", default=None)
ap.add_argument("--out", default=None)
# ablations only
ap.add_argument("--report_in_context", action="store_true", help="show the generated report before each question")
ap.add_argument("--single_question", action="store_true", help="use only the first surviving phrasing per type")
args = ap.parse_args()

QUESTIONS = {
    "effusion":     ["Is there a pleural effusion in this chest X-ray?",
                     "Does this chest X-ray show fluid in the pleural space?",
                     "Is a pleural effusion present on this radiograph?"],
    "pneumothorax": ["Is there a pneumothorax in this chest X-ray?",
                     "Does this chest X-ray show air in the pleural space?",
                     "Is a pneumothorax present on this radiograph?"],
    "opacity":      ["Is there an area of increased opacity in the lungs on this chest X-ray?",
                     "Is there consolidation in the lungs on this chest X-ray?",
                     "Is there airspace disease in the lungs on this radiograph?",
                     "Does this chest X-ray show an airspace opacity?",
                     "Is part of the lung whiter or denser than normal on this chest X-ray?"],
    "atelectasis":  ["Is there atelectasis in this chest X-ray?",
                     "Does this chest X-ray show collapse or volume loss of part of the lung?",
                     "Is atelectasis present on this radiograph?"],
    "edema":        ["Is there pulmonary edema in this chest X-ray?",
                     "Does this chest X-ray show fluid in the lungs consistent with edema?",
                     "Is pulmonary edema present on this radiograph?"],
    "interstitial": ["Are there increased interstitial markings in this chest X-ray?",
                     "Does this chest X-ray show an interstitial lung pattern?",
                     "Are the interstitial lung markings abnormally prominent on this radiograph?"],
    "congestion":   ["Is there pulmonary vascular congestion in this chest X-ray?",
                     "Are the pulmonary vessels engorged on this chest X-ray?",
                     "Does this radiograph show increased pulmonary vascularity?"],
    "cardiomegaly": ["Is the heart enlarged in this chest X-ray?",
                     "Is there cardiomegaly on this chest X-ray?",
                     "Is the cardiac silhouette enlarged on this radiograph?"],
    "nodule":       ["Is there a lung nodule or mass in this chest X-ray?",
                     "Does this chest X-ray show a nodule in the lungs?",
                     "Is a pulmonary nodule or mass present on this radiograph?"],
    "fracture":     ["Is there a fracture in this chest X-ray?",
                     "Does this chest X-ray show a broken rib or other fracture?",
                     "Is a fracture present on this radiograph?"],
    "granuloma":    ["Is there a calcified granuloma in this chest X-ray?",
                     "Does this chest X-ray show a calcified granuloma?",
                     "Is a calcified granuloma present on this radiograph?"],
    "degenerative": ["Are there degenerative changes of the spine in this chest X-ray?",
                     "Does this chest X-ray show spondylosis or osteophytes?",
                     "Is there degenerative change of the thoracic spine on this radiograph?"],
    "surgical":     ["Are there surgical clips, wires or other metallic objects visible on this chest X-ray?",
                     "Does this radiograph show surgical hardware or a metallic foreign body?",
                     "Are there metallic surgical clips or sternal wires on this chest X-ray?"],
    "lines":        ["Is there a catheter, line or tube visible on this chest X-ray?",
                     "Does this chest X-ray show a central venous catheter, PICC line or port?",
                     "Is there a medical line, tube or pacemaker lead on this radiograph?"],
    "calcification": ["Is there calcification of the aorta on this chest X-ray?",
                      "Does this chest X-ray show vascular calcification?",
                      "Are there calcified structures visible on this radiograph?"],
    "scoliosis":    ["Is there scoliosis on this chest X-ray?",
                     "Is the thoracic spine abnormally curved on this chest X-ray?",
                     "Does this radiograph show a curvature of the spine?"],
    "emphysema":    ["Are the lungs hyperinflated on this chest X-ray?",
                     "Does this chest X-ray show emphysema?",
                     "Are the hemidiaphragms flattened on this radiograph?"],
    "hilar":        ["Are the hila enlarged or prominent on this chest X-ray?",
                     "Is there hilar enlargement on this chest X-ray?",
                     "Does this radiograph show prominent hilar structures?"],
    "hemidiaphragm": ["Is one of the hemidiaphragms elevated on this chest X-ray?",
                      "Is there elevation of the diaphragm on this chest X-ray?",
                      "Does this radiograph show an elevated hemidiaphragm?"],
}
SUFFIX = " Answer only yes or no."

# Data
if args.dataset == "chexpert":
    from chexpert_data import get_chexpert
    ds = get_chexpert()
else:
    from iu_xray_data import get_dataset
    ds = get_dataset("iu_xray", img_root=args.img_root)
split_ids = {str(ds[i].get("id", i)) for i in json.load(open(args.split_file))[args.split_name]}
by_id = {str(s.get("id", i)): s for i, s in enumerate(ds)}
preds = {}
for path in args.preds.split(","):
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r["id"] in split_ids:
                preds[r["id"]] = r
ids = sorted(preds)
print(f"{args.mode}: {len(ids)} reports from {args.preds} in split '{args.split_name}'")

# Model
DEFAULT_MODELS = {"medgemma": "google/medgemma-4b-it", "chexagent": "StanfordAIMI/CheXagent-8b",
                  "llavamed": "chaoyinshe/llava-med-v1.5-mistral-7b-hf"}
adapter = {"medgemma": MedGemmaAdapter, "chexagent": CheXagentAdapter, "llavamed": LlavaMedAdapter}[args.backbone]()
adapter.load(args.model or DEFAULT_MODELS[args.backbone], bf16=True, debug=False)
tok = adapter.processor.tokenizer
print(f"backbone: {args.backbone}")


def first_ids(words):
    # first token ids that decode to an answer word (skips bare pieces like SentencePiece's '▁')
    targets = {w.lower() for w in words}
    out = set()
    for w in words:
        for v in (w, " " + w):
            for tid in tok.encode(v, add_special_tokens=False)[:2]:
                if tok.decode([tid]).strip().lower() in targets:
                    out.add(tid)
                    break
    return sorted(out)


YES, NO = first_ids(["Yes", "yes", "YES"]), first_ids(["No", "no", "NO"])
assert YES and NO and not set(YES) & set(NO), f"bad answer tokens: yes={YES} no={NO}"


@torch.no_grad()
def p_yes(image, q, context=None):
    # P(yes) / (P(yes) + P(no)) for the first answer token, one forward pass
    prompt = q + SUFFIX
    if context:
        prompt = f'Draft report for this chest X-ray: "{context}"\n\n' + prompt
    logits, _, _ = adapter.forward_step(inputs=adapter.build_inputs(image, prompt), past_key_values=None)
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    py, pn = lp[YES].exp().sum().item(), lp[NO].exp().sum().item()
    return py / (py + pn + 1e-12)


blank_cache = {}


def blank_p_yes(q, n):
    # bias filter: P(yes) on a black and a grey image (n views), the larger of the two
    if (q, n) not in blank_cache:
        vals = []
        for color in ((0, 0, 0), (128, 128, 128)):
            b = Image.new("RGB", (896, 896), color)
            vals.append(p_yes([b] * n if n > 1 else b, q))
        blank_cache[(q, n)] = max(vals)
    return blank_cache[(q, n)]


def recheck(image, g, report=None):
    n = len(image) if isinstance(image, list) else 1
    qs = [q for q in QUESTIONS[g] if blank_p_yes(q, n) <= args.prior_max]
    if args.single_question:
        qs = qs[:1]
    if not qs:
        return {"group": g, "questions": [], "p_yes": [], "score": None, "action": "unverifiable"}
    ps = [p_yes(image, q, report if args.report_in_context else None) for q in qs]
    return {"group": g, "questions": qs, "p_yes": ps, "score": sum(ps) / len(ps), "action": "scored"}


@torch.no_grad()
def token_entropies(image, prompt, text):
    # one teacher-forced pass over the report with the clean image; normalized entropy per token
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    gen = enc.input_ids
    if not gen:
        return [], []
    inputs = adapter.build_inputs(image, prompt)
    g = torch.tensor([gen], device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)
    full = dict(inputs)
    full["input_ids"] = torch.cat([inputs["input_ids"], g], dim=1)
    if "attention_mask" in inputs:
        full["attention_mask"] = torch.ones_like(full["input_ids"])
    if "token_type_ids" in inputs:
        full["token_type_ids"] = torch.cat([inputs["token_type_ids"], torch.zeros_like(g)], dim=1)
    n = len(gen)
    # index from the end: CheXagent prepends visual query tokens
    lc = adapter.model(**full).logits[0, -(n + 1):-1].float()
    lp = F.log_softmax(lc, -1)
    ent = (-(lp.exp() * lp).sum(-1) / math.log(lc.shape[-1])).tolist()
    return ent, [tuple(o) for o in enc.offset_mapping]


def claim_entropy(text, g, ent, offs):
    # mean token entropy over the first sentence that mentions finding type g
    low = text.lower()
    for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
        s = low[m.start():m.end()]
        if any(k in s for k in K[g]) and (g != "hemidiaphragm" or "diaphragm" in s):
            idx = [i for i, (a, b) in enumerate(offs) if a >= m.start() and b <= m.end()]
            return sum(ent[i] for i in idx) / len(idx) if idx else None
    return None


def check_report(r, image):
    results = [recheck(image, g, r["prediction"]) for g in affirmed_groups(r["prediction"])]
    extra = 0
    if any(v["action"] != "unverifiable" for v in results):
        ent, offs = token_entropies(image, r.get("prompt", ""), r["prediction"])
        extra = 1
        for v in results:
            if v["action"] != "unverifiable":
                v["entropy"] = claim_entropy(r["prediction"], v["group"], ent, offs)
    return results, extra


# run (resumable)
if args.mode == "calibrate":
    out = args.out or f"results/verify_{args.tag}_{args.split_name}_rows.jsonl"
else:
    out = args.out or f"results/{args.tag}_verified_p{args.prompt_id}_{args.split_name}.jsonl"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
done = {json.loads(l)["id"] for l in open(out) if l.strip()} if os.path.exists(out) else set()
if done:
    print(f"resuming: {len(done)} reports already done")

with open(out, "a") as f:
    for rid in tqdm(ids, desc=f"verify/{args.tag}/{args.split_name}"):
        if rid in done:
            continue
        r, image = preds[rid], by_id[rid]["image"]
        t0 = time.perf_counter()
        results, extra_passes = check_report(r, image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        n_forward = sum(len(v["p_yes"]) for v in results) + extra_passes
        if args.mode == "calibrate":
            for v in results:
                v.update(id=rid, ref_status=ref_support(r["ground_truth"], v["group"]) or "neg")
                f.write(json.dumps(v) + "\n")
            if not results:  # mark the report as done
                f.write(json.dumps({"id": rid, "group": None}) + "\n")
        else:
            f.write(json.dumps({"id": rid, "backbone": args.backbone, "mode": f"{args.tag}+verify",
                                "prompt": r.get("prompt", ""), "prediction": r["prediction"],
                                "original_prediction": r["prediction"], "ground_truth": r["ground_truth"],
                                "verification": results, "seconds": round(seconds, 4),
                                "n_forward": n_forward}) + "\n")
        f.flush()

# Summary
print("\nblank-image P(yes) per question (above --prior_max = not used):")
for (q, n), p in sorted(blank_cache.items(), key=lambda x: -x[1]):
    print(f"  {p:.3f} {'DROPPED' if p > args.prior_max else 'kept   '} {q}")

rows = [json.loads(l) for l in open(out) if l.strip()]
if args.mode == "apply":
    secs = [r["seconds"] for r in rows]
    fwd = [r["n_forward"] for r in rows]
    print(f"\n{len(rows)} reports; {sum(secs) / len(secs):.3f} s/report, {sum(fwd) / len(fwd):.2f} forward passes/report")
    print(f"-> {out}")
    sys.exit(0)

C = [r for r in rows if r.get("group") and r["action"] != "unverifiable" and r["ref_status"] in ("pos", "neg")]


def auc(rs):
    s = [r["score"] for r in rs if r["ref_status"] == "pos"]
    h = [r["score"] for r in rs if r["ref_status"] == "neg"]
    if not s or not h:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in s for b in h) / (len(s) * len(h))


rng = random.Random(0)
bs = sorted(auc(rng.choices(C, k=len(C))) for _ in range(1000))
n_sup = sum(r["ref_status"] == "pos" for r in C)
print(f"\nre-checked claims: {len(C)} (supported {n_sup}, unsupported {len(C) - n_sup}); "
      f"unverifiable {sum(r.get('action') == 'unverifiable' for r in rows)}")
print(f"re-check AUROC = {auc(C):.3f}  95% CI [{bs[25]:.3f}, {bs[975]:.3f}]")
print("per finding type:")
for g in K:
    rs = [r for r in C if r["group"] == g]
    if rs:
        print(f"  {g:13} supported={sum(r['ref_status'] == 'pos' for r in rs):3} "
              f"unsupported={sum(r['ref_status'] == 'neg' for r in rs):3}  AUROC={auc(rs):.3f}")
print(f"\nrows -> {out}")