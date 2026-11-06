import io
import os
import json
from PIL import Image
from datasets import load_dataset

KNOWN_DATASET_KEYWORDS = {"iu_xray", "iu-xray", "iu_xray_rrg", "iu-xray-rrg"}

def parse_hf_sample(sample, image_field="main_image", text_field="findings_section"): # Extracts and standardizes PIL Image and text ground truth from a HuggingFace sample.
    if image_field not in sample:
        raise KeyError(
            f"Expected image field '{image_field}' not found in HF sample. "
            f"Available fields: {list(sample.keys())}"
        )
    img = sample[image_field]
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
    elif isinstance(img, dict) and "bytes" in img:
        img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    else:
        img = Image.open(img).convert("RGB")

    if text_field not in sample:
        raise KeyError(
            f"Expected text field '{text_field}' not found in HF sample. "
            f"Available fields: {list(sample.keys())}. "
            "Refusing to silently default to an empty ground truth."
        )
    ground_truth = sample[text_field]
    if ground_truth is None or not str(ground_truth).strip():
        raise ValueError(
            "Ground-truth findings_section is empty for this sample. If the "
            "dataset is expected to have some empty-findings reports, filter "
            "them out explicitly upstream rather than passing them through "
            "silently -- an empty reference will silently zero out or distort "
            "several of the metrics computed later."
        )
    return img, ground_truth


def load_iu_xray_data(split="test"): # Loads and standardizes the IU X-Ray dataset from HuggingFace
    print(f"Loading IU X-Ray dataset ({split} split)...")
    hf_dataset = load_dataset("X-iZhang/IU-Xray-RRG", "findings_section", split=split)

    dataset = []
    skipped = []
    for idx, sample in enumerate(hf_dataset):
        try:
            image, ground_truth = parse_hf_sample(sample)
        except (KeyError, ValueError) as e:
            skipped.append({"idx": idx, "error": str(e)})
            continue
        dataset.append({
            "id": f"iu_xray_{idx}",
            "image": image,  # Holds PIL Image directly
            "prompt": "Describe the detailed medical findings in this chest X-ray.",
            "ground_truth": ground_truth
        })

    if skipped:
        print(f"WARNING: skipped {len(skipped)} HF samples due to missing/empty fields. "
              f"First few: {skipped[:5]}")
        print("If this number is large, STOP and check image_field/text_field against "
              "the actual dataset schema before proceeding -- do not assume the field "
              "names guessed here are correct without checking.")
    print(f"Loaded {len(dataset)} samples from IU X-Ray ({len(skipped)} skipped).")
    return dataset


def load_local_json_data(json_path, img_root): # Loads and standardizes a local JSON dataset.
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON path not found: {json_path}")
    if img_root is None:
        raise ValueError(
            "img_root is required for load_local_json_data and was None. "
            "Pass the directory containing the images referenced by this JSON."
        )
    if not os.path.exists(img_root):
        raise FileNotFoundError(f"img_root not found: {img_root}")

    print(f"Loading local JSON dataset from: {json_path}...")
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    dataset = []
    skipped = []
    for sample in raw_data:
        img_path = os.path.join(img_root, sample["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            skipped.append({"id": sample.get("id", sample["image"]), "path": img_path, "error": str(e)})
            continue

        if "conversations" in sample and len(sample["conversations"]) > 0:
            prompt = sample["conversations"][0]["value"]
            prompt = prompt.replace("<image>", "").replace("<|image_1|>", "").strip()
        else:
            prompt = "Describe the detailed medical findings in this chest X-ray."

        ground_truth = sample.get("ground_truth", "")
        if not ground_truth or not str(ground_truth).strip():
            skipped.append({"id": sample.get("id", sample["image"]), "path": img_path,
                             "error": "empty ground_truth"})
            continue

        dataset.append({
            "id": sample.get("id", sample["image"]),
            "image": image,
            "prompt": prompt,
            "ground_truth": ground_truth
        })

    if skipped:
        print(f"WARNING: skipped {len(skipped)} local samples. First few: {skipped[:5]}")
    print(f"Loaded {len(dataset)} samples from local JSON ({len(skipped)} skipped).")
    return dataset


def get_dataset(dataset_name_or_path, img_root=None, split="test"): # Unified Dataset
    normalized = dataset_name_or_path.strip().lower()

    if normalized in KNOWN_DATASET_KEYWORDS:
        return load_iu_xray_data(split=split)

    if not os.path.exists(dataset_name_or_path):
        raise ValueError(
            f"'{dataset_name_or_path}' is not a recognized dataset keyword "
            f"(expected one of {sorted(KNOWN_DATASET_KEYWORDS)}) and is not an "
            "existing local file path. Pass exactly one of the known keywords, "
            "or a valid path to a local JSON file."
        )

    if img_root is None:
        raise ValueError(
            f"img_root must be provided when loading a local JSON dataset "
            f"(got dataset_name_or_path='{dataset_name_or_path}', img_root=None)."
        )

    return load_local_json_data(json_path=dataset_name_or_path, img_root=img_root)