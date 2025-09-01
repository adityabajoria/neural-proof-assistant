import snorkel
import os
import re
import numpy as np
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from math_subjects import get_subject_lfs
from proof_tactics import get_tactic_lfs

DATA_PATH = "/Users/aditya/neural-proof-assistant/backend/data/unlabeled_data/unlabeled_goals.csv"
OUTPUT_DIR = "/Users/aditya/neural-proof-assistant/backend/data/weak_labels"
TEXT_COL = "text"

N_SUBJECTS = 12
N_TACTICS = 7

ABSTAIN = -1

SUBJECT_THRES = 0.6
TACTIC_THRES = 0.6

SEED = 123
os.makedirs(OUTPUT_DIR, exist_ok=True)


data = pd.read_csv(DATA_PATH)
print(f"Loaded {len(data)} samples")
data[TEXT_COL] = data[TEXT_COL].fillna("").astype(str)

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = (s.replace("∃", " there exists ")
        .replace("\\exists", " there exists ")
        .replace("∀", " for all ")
        .replace("\\forall", " for all ")
        .replace("≠", " not equal ")
        .replace("→", " implies ")
        .replace("⇒", " implies ")
        .replace("^c", " complement ")
        .replace("∩", " intersection ")
        .replace("∪", " union "))
    s = s.strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    replacements = {
        "∃": " there exists ", "\\exists": " there exists ",
        "∀": " for all ", "\\forall": " for all ",
        "≠": " not equal ", "!=": " not equal ",
        "→": " implies ", "⇒": " implies ", "\\implies": " implies ",
        "↔": " iff ", "⟺": " iff ", "\\iff": " iff ", "<->": " iff ",
        "¬": " not ", "\\neg": " not ",
        "\\wlog": " without loss of generality ",
        "∧": " and ", "\\wedge": " and ", "∨": " or ", "\\vee": " or ",
        "∴": " therefore ", "\\therefore": " therefore ",
        "->": " implies ", "=>": " implies ",
        "q.e.d.": " qed ", "□": " qed ", "\\qed": " qed "
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s
data['text'] = data['text'].apply(clean_text)

# SUBJECT LFS
SUBJECT_LFS = get_subject_lfs()
print("Loaded Subject LFs:", [lf.name for lf in SUBJECT_LFS])

applier = PandasLFApplier(lfs=SUBJECT_LFS)
L_subject = applier.apply(df=data)

print("Label matrix shape:", L_subject.shape)
print("Label matrix values:\n", L_subject)

# TACTIC LFS
TACTIC_LFS = get_tactic_lfs()
print("Loaded Tactic LFs:", [lf.name for lf in TACTIC_LFS])

applier = PandasLFApplier(lfs=TACTIC_LFS)
L_tactics = applier.apply(df=data)

print("Label matrix shape", L_tactics.shape)
print("Label matrix values:\n", L_tactics)

print("\n[Subjects]")
analysis = LFAnalysis(L=L_subject, lfs=SUBJECT_LFS)
print(analysis.lf_summary())

print("\n[Tactics]")
analysis_tactics = LFAnalysis(L=L_tactics, lfs=TACTIC_LFS)
print(analysis_tactics.lf_summary())