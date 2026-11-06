# Finding-level precision / recall / F1 / unsupported-finding rate from RadGraph entities.
# Reads the RadGraph annotations RadEval already returns (do_radgraph=True, do_details=True):
#   raw["radgraph"]["hypothesis_annotation_lists"], raw["radgraph"]["reference_annotation_lists"]
# A finding is a head Observation entity (one that does not "modify" another entity), with its assertion.
import re
from collections import defaultdict

# normal-descriptor words and cardiomegaly synonyms are mapped to one shared term each,
# so different wording of the same statement still matches
NORMAL_DESCRIPTOR_TERMS = {
    "unremarkable", "clear", "intact", "normal", "normallimits", "normal limits", "withinnormal",
    "tonormal", "within", "limits", "normalsize", "well defined", "well expanded", "well aerated",
    "well aligned", "well visualized",
}
NORMAL_DESCRIPTOR_CANONICAL = "normal_descriptor"
CARDIOMEGALY_CLUSTER = {
    "enlarged", "enlargement", "enlarge", "cardiomegaly",
    "cardiomegally", "cardiomegly", "cardiomegy",  # misspellings seen in RadGraph output
}
CARDIOMEGALY_CANONICAL = "cardiomegaly"


def _canonicalize_finding_term(term):
    if term in NORMAL_DESCRIPTOR_TERMS:
        return NORMAL_DESCRIPTOR_CANONICAL
    if term in CARDIOMEGALY_CLUSTER:
        return CARDIOMEGALY_CANONICAL
    return term


def _normalize_assertion(label):
    # "Observation::definitely present" -> "present"; Anatomy and unknown labels -> None
    if "::" not in label:
        return None
    entity_type, assertion = label.split("::", 1)
    if entity_type.strip() != "Observation":
        return None
    assertion = assertion.strip().lower()
    if "present" in assertion:
        return "present"
    if "absent" in assertion:
        return "absent"
    if "uncertain" in assertion:
        return "uncertain"
    return None


def _normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def entities_to_findings(entities, mode="head_only"):
    # set of (finding, assertion); head_only (used in the paper) keys on the head word,
    # full_phrase also attaches the modifier words (stricter)
    modifiers_of = defaultdict(list)
    has_outgoing_modify = set()
    for eid, ent in entities.items():
        for rel in ent.get("relations", []):
            if not rel or len(rel) != 2:
                continue
            rel_type, target_id = rel
            if rel_type == "modify":
                modifiers_of[str(target_id)].append(eid)
                has_outgoing_modify.add(eid)

    findings = set()
    for eid, ent in entities.items():
        assertion = _normalize_assertion(ent.get("label", ""))
        if assertion is None or eid in has_outgoing_modify:
            continue
        if mode == "full_phrase":
            parts = [(entities[m]["start_ix"], entities[m]["tokens"]) for m in modifiers_of.get(eid, []) if m in entities]
            parts.append((ent["start_ix"], ent["tokens"]))
            key_text = " ".join(tok for _, tok in sorted(parts, key=lambda x: x[0]))
        else:
            key_text = ent["tokens"]
        findings.add((_canonicalize_finding_term(_normalize_text(key_text)), assertion))
    return findings


def annotation_list_to_findings(annotation_list, mode="head_only"):
    return [entities_to_findings(entry.get("entities", {}), mode=mode) for entry in annotation_list]


def compute_confusion(generated_set, reference_set):
    # only "present" findings count; correct negations are true negatives and are not scored
    gen_present = {e for e, a in generated_set if a == "present"}
    ref_present = {e for e, a in reference_set if a == "present"}
    return len(gen_present & ref_present), len(gen_present - ref_present), len(ref_present - gen_present)


def _aggregate(hyp_findings, ref_findings):
    total_tp = total_fp = total_fn = 0
    per_sample = []
    for i, (gset, rset) in enumerate(zip(hyp_findings, ref_findings)):
        tp, fp, fn = compute_confusion(gset, rset)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_sample.append({"index": i, "tp": tp, "fp": fp, "fn": fn})

    nan = float("nan")
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else nan
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else nan
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and precision == precision and recall == recall else nan)
    return {
        "n_samples": len(hyp_findings),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": total_fp / (total_tp + total_fp) if total_tp + total_fp else nan,  # H = 1 - P
        "per_sample": per_sample,
    }


def compute_finding_level_metrics(raw, modes=("head_only", "full_phrase")):
    hyp_lists = raw["radgraph"]["hypothesis_annotation_lists"]
    ref_lists = raw["radgraph"]["reference_annotation_lists"]
    assert len(hyp_lists) == len(ref_lists), f"{len(hyp_lists)} hypotheses vs {len(ref_lists)} references"
    return {mode: _aggregate(annotation_list_to_findings(hyp_lists, mode), annotation_list_to_findings(ref_lists, mode))
            for mode in modes}


if __name__ == "__main__":
    # quick check, no model needed
    ents = {
        "1": {"tokens": "obvious", "label": "Observation::definitely absent", "start_ix": 0, "relations": [["modify", "2"]]},
        "2": {"tokens": "abnormalities", "label": "Observation::definitely absent", "start_ix": 1, "relations": []},
        "3": {"tokens": "chest", "label": "Anatomy::definitely present", "start_ix": 2, "relations": []},
        "4": {"tokens": "cardiomegaly", "label": "Observation::definitely present", "start_ix": 3, "relations": []},
        "5": {"tokens": "enlarged", "label": "Observation::definitely present", "start_ix": 4, "relations": []},
        "6": {"tokens": "intact", "label": "Observation::definitely present", "start_ix": 5, "relations": []},
    }
    got = entities_to_findings(ents)
    assert got == {("abnormalities", "absent"), ("cardiomegaly", "present"), ("normal_descriptor", "present")}, got
    print("ok:", sorted(got))