# Greedy report generation with a findings-only prompt, for one split at a time.

import argparse
import json
import os
import re
import statistics as st
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models as _cedar
from models import vanilla_greedy_generate

PROMPTS = {
    1: "Write the findings section of a radiology report for this chest X-ray in 3 to 5 sentences.",
    2: "Describe the findings in this chest X-ray as a radiologist would in the findings section of a report. Be concise.",
    3: "Write a concise findings section for this chest X-ray. Do not include an introduction, headings, or disclaimers.",
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="iu_xray", choices=["iu_xray", "chexpert"])
ap.add_argument("--split_file", required=True)
ap.add_argument("--split_name", default="cal", choices=["dev", "cal", "test"])
ap.add_argument("--prompt_id", type=int, default=1, choices=sorted(PROMPTS))
ap.add_argument("--pilot", type=int, default=0,
                help="N > 0: run every candidate prompt on the first N cal reports, print, and exit")
ap.add_argument("--backbone", default=os.environ.get("BACKBONE", "medgemma"),
                choices=["medgemma", "chexagent", "llavamed"], help="defaults to $BACKBONE, else medgemma")
ap.add_argument("--model", default=None, help="checkpoint; default depends on --backbone")
ap.add_argument("--img_root", default=None)
ap.add_argument("--max_new_tokens", type=int, default=256)
ap.add_argument("--repetition_penalty", type=float, default=1.25)
ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
ap.add_argument("--out", default=None, help="default: results/greedy_findings_p<id>_<split>.jsonl")
args = ap.parse_args()

# Data
if args.dataset == "chexpert":
    from chexpert_data import get_chexpert
    ds = get_chexpert()
else:
    from iu_xray_data import get_dataset
    ds = get_dataset("iu_xray", img_root=args.img_root)
split_name = "cal" if args.pilot else args.split_name
idx = json.load(open(args.split_file))[split_name]
samples = [ds[i] for i in idx]
print(f"split={split_name}: {len(samples)} reports")

# Model (bf16)
ADAPTERS = {"medgemma": "MedGemmaAdapter", "chexagent": "CheXagentAdapter", "llavamed": "LlavaMedAdapter"}
DEFAULT_MODELS = {"medgemma": "google/medgemma-4b-it", "chexagent": "StanfordAIMI/CheXagent-8b",
                  "llavamed": "chaoyinshe/llava-med-v1.5-mistral-7b-hf"}
args.model = args.model or DEFAULT_MODELS[args.backbone]
if not hasattr(_cedar, ADAPTERS[args.backbone]):
    sys.exit(f"{ADAPTERS[args.backbone]} not found in {_cedar.__name__} -- this method does not support "
             f"backbone {args.backbone!r} yet")
print(f"backbone: {args.backbone}  model: {args.model}")
adapter = getattr(_cedar, ADAPTERS[args.backbone])()
adapter.load(args.model, bf16=True, debug=False)
tok = adapter.processor.tokenizer


def generate(sample, prompt):
    with torch.no_grad():
        pred, _ = vanilla_greedy_generate(
            adapter, sample["image"], prompt, max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    n_tok = len(tok(pred, add_special_tokens=False).input_ids)
    return pred, n_tok


def looks_chatty(t):
    return bool(re.match(r"\s*(okay|sure|here)", t, re.I)) or "**" in t or "#" in t


# Compare prompts
if args.pilot:
    pilot = samples[: args.pilot]
    for pid, prompt in PROMPTS.items():
        lens, chatty, cut = [], 0, 0
        print("\n" + "=" * 90 + f"\nPROMPT {pid}: {prompt}\n" + "=" * 90)
        for i, s in enumerate(pilot):
            pred, n = generate(s, prompt)
            lens.append(n)
            chatty += looks_chatty(pred)
            cut += n >= args.max_new_tokens - 2
            if i < 4:
                print(f"\n[{s.get('id')}] ({n} tokens)\n  MODEL: {pred[:500]}\n  REF:   {s.get('ground_truth', '')[:300]}")
        print(f"\nsummary prompt {pid}: median {st.median(lens):.0f} tokens, max {max(lens)}, "
              f"chatty/markdown {chatty}/{len(pilot)}, hit token limit {cut}/{len(pilot)}")
    print("\nPick the prompt with short, clean findings (no opener/markdown, nothing cut off), "
          "then run with --split_name cal and test using --prompt_id <n>.")
    sys.exit(0)

# Run
prompt = PROMPTS[args.prompt_id]
out = args.out or f"results/greedy_findings_p{args.prompt_id}_{args.split_name}.jsonl"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
done = set()
if os.path.exists(out):
    done = {json.loads(l)["id"] for l in open(out) if l.strip()}
    print(f"resuming: {len(done)} already done")

with open(out.replace(".jsonl", "_config.json"), "w") as f:
    json.dump({"prompt_id": args.prompt_id, "prompt": prompt, "system_prompt": getattr(adapter, "system_prompt", None),
               "model": args.model, "dtype": "bf16", "decoding": "greedy",
               "max_new_tokens": args.max_new_tokens, "repetition_penalty": args.repetition_penalty,
               "no_repeat_ngram_size": args.no_repeat_ngram_size, "split": args.split_name}, f, indent=2)

n_cut = 0
with open(out, "a") as f:
    for s in tqdm(samples, desc=f"greedy/{args.split_name}"):
        sid = str(s.get("id"))
        if sid in done:
            continue
        pred, n = generate(s, prompt)
        n_cut += n >= args.max_new_tokens - 2
        f.write(json.dumps({"id": sid, "backbone": args.backbone, "mode": "greedy", "prompt": prompt,
                            "prediction": pred, "ground_truth": s.get("ground_truth", ""),
                            "n_tokens": n}) + "\n")
        f.flush()

print(f"\ndone -> {out}  ({n_cut} reports hit the {args.max_new_tokens}-token limit this run)")