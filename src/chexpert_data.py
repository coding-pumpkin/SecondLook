# CheXpert Plus loader with the same interface as the IU X-Ray loader: a list of samples where
import json
import os

from PIL import Image

ROOT = os.environ.get("CHEXPERT_DIR", "data/chexpert")  # folder with chexpert_subset.jsonl


class Sample(dict):
    """dict whose "image" entry is opened from disk on access."""

    def __getitem__(self, k):
        if k == "image":
            return Image.open(dict.__getitem__(self, "image_path")).convert("RGB")
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default


def get_chexpert(root=ROOT):
    path = os.path.join(root, "chexpert_subset.jsonl")
    out = []
    for line in open(path):
        r = json.loads(line)
        s = Sample(r)
        dict.__setitem__(s, "image", None)  # placeholder so "image" in s is True; loaded on access
        # reference = findings section; exposed under the keys different scripts may use
        for k in ("report", "ground_truth", "gt", "text", "caption"):
            dict.__setitem__(s, k, r["findings"])
        out.append(s)
    print(f"Loaded {len(out)} samples from CheXpert Plus subset ({path})")
    return out