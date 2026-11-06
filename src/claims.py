# Claim extraction, reference labelling and clause-level deletion, shared by all SecondLook scripts.
import re

# 19 finding types and the keywords that mention them
K = {
    "effusion":     ["effusion"],
    "pneumothorax": ["pneumothora"],
    "opacity":      ["opacit", "consolidat", "airspace", "infiltrat", "pneumonia"],
    "atelectasis":  ["atelecta"],
    "edema":        ["edema"],
    "interstitial": ["interstitial", "reticular"],
    "congestion":   ["congestion", "vascular engorgement", "cephalization"],
    "cardiomegaly": ["cardiomegaly", "enlarg"],
    "nodule":       ["nodul", "mass"],
    "fracture":     ["fractur"],
    "granuloma":    ["granulom", "calcified nodule"],
    "degenerative": ["degenerat", "spondyl", "osteophyt"],
    # frequent false-positive types
    "surgical":     ["surgical clip", "clips", "sternal wire", "wires", "sternotomy", "hardware", "fixation",
                     "foreign body", "metallic", "radiopaque"],
    "lines":        ["catheter", "picc", "central line", "venous line", "port a cath", "port-a-cath", "pacemaker",
                     "enteric tube", "endotracheal tube", "chest tube", "feeding tube", "ng tube"],
    "calcification": ["calcification", "calcified aort", "atherosclero"],
    "scoliosis":    ["scolio", "curvature of the spine", "spinal curvature"],
    "emphysema":    ["emphysema", "hyperinflat", "hyperexpan", "flattened diaphragm", "flattening of the"],
    "hilar":        ["hilar", "hilum", "hila "],
    "hemidiaphragm": ["elevat"],  # also needs "diaphragm", see mentions()
}
# a claim is supported if the reference affirms the finding or one of these equivalents
SUPPORT_ALIASES = {"nodule": ["granuloma"], "granuloma": ["nodule"], "opacity": ["atelectasis"]}

NEG = re.compile(r"\b(no|without|negative|not|normal|clear|unremarkable|free|resolved)\b")
UNC = re.compile(r"upper limits? of normal|borderline|top (of )?normal|upper normal|high normal")
HEART = re.compile(r"heart|cardiac|cardiomediastinal|cardiomegaly")
DIFF = re.compile(r"could be due to|due to various|such as|including|rule out|differential|may not represent")
VERB = re.compile(r"\b(is|are|appears?|seen|noted|shows?|demonstrat\w*|present)\b")
CONTRAST = re.compile(r"\b(but|however|although)\b")
DEPENDENT = re.compile(r"^(which|that|suggesting|likely|possibly|probably|could|may|might|consistent with|representing)\b")
HILAR_ABNORMAL = ("prominen", "enlarg", "fullness", "adenopathy", "lymphadenopath", "mass", "bulk")


def mentions(g, c):
    if g == "cardiomegaly":
        return bool(HEART.search(c)) and ("enlarg" in c or "cardiomegaly" in c or bool(UNC.search(c)))
    if g == "hemidiaphragm":
        return "diaphragm" in c and "elevat" in c
    if g == "hilar":
        # "hilar contours are stable" is not a claim; needs an abnormality word
        return any(k in c for k in K[g]) and any(a in c for a in HILAR_ABNORMAL)
    return any(k in c for k in K[g])


def split_sentences(text):
    return [p for p in re.split(r"(?<=[.!?;])\s+|\n+", text.strip()) if p.strip()]


def clause_labels(sentence):
    # label each finding in each clause: pos / neg / unc (borderline) / diff (differential)
    # negation wins, and carries over to following verbless clauses until "but/however/although"
    out, carried_neg, carried_diff = [], False, False
    for raw in [c for c in re.split(r",\s*", sentence) if c.strip()]:
        c = re.sub(r"[*#_`]", " ", raw).lower().strip()
        own_neg, own_diff, verbless = bool(NEG.search(c)), DIFF.search(c), not VERB.search(c)
        neg = own_neg or (carried_neg and verbless)
        labels = {}
        for g in K:
            if not mentions(g, c):
                continue
            if neg:
                labels[g] = "neg"
                continue
            if own_diff:
                pos = [c.find(k) for k in K[g] if c.find(k) >= 0]
                if pos and min(pos) > own_diff.start():
                    labels[g] = "diff"
                    continue
            elif carried_diff and verbless:
                labels[g] = "diff"
                continue
            labels[g] = "unc" if UNC.search(c) else "pos"
        if CONTRAST.search(c):
            carried_neg = carried_diff = False
        if own_neg:
            carried_neg = True
        if own_diff:
            carried_diff = True
        out.append((raw, labels))
    return out


def report_status(text, g):
    st = {lab.get(g) for s in split_sentences(text) for _, lab in clause_labels(s)} - {None}
    for s in ("pos", "diff", "unc", "neg"):
        if s in st:
            return s
    return None


def affirmed_groups(text):
    return [g for g in K if report_status(text, g) == "pos"]


def ref_support(text, g):
    # reference status of finding g, counting equivalent findings
    sts = [report_status(text, x) for x in [g] + SUPPORT_ALIASES.get(g, [])]
    for s in ("pos", "unc", "diff", "neg"):
        if s in sts:
            return s
    return None


def edit_report(text, actions):
    # actions: {group: "drop" | "keep"}; removes the clauses of dropped findings
    new_sents = []
    for sent in split_sentences(text):
        cl = clause_labels(sent)
        keep = []
        for raw, labels in cl:
            pos = [g for g, s in labels.items() if s == "pos"]
            acts = [actions.get(g, "keep") for g in pos]
            keep.append((not pos) or any(a in ("keep", "unverifiable") for a in acts))
        for i in range(1, len(cl)):
            # a dependent clause ("which could represent ...") goes with its removed head
            if not keep[i - 1] and DEPENDENT.match(cl[i][0].strip().lower()):
                keep[i] = False
        kept = [cl[i][0].strip() for i in range(len(cl)) if keep[i]]
        if not kept:
            continue
        if all(keep):
            new_sents.append(sent.strip())
            continue
        s = re.sub(r"^(and|or|but|with|also|as well as)\s+", "", ", ".join(kept), flags=re.I)
        s = s[0].upper() + s[1:]
        new_sents.append(s if re.search(r"[.!?]$", s) else s + ".")
    return " ".join(new_sents) if new_sents else text.strip()