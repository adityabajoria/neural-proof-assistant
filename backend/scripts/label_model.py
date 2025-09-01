from backend.weaklabel_pipeline import data, L_subject, L_tactics
import numpy as np
import os
from snorkel.labeling.model import LabelModel

ABSTAIN = -1
SUBJECT_CARD = 12
TACTIC_CARD = 7

SUBJECT_THRES = 0.4
TACTIC_THRES = 0.4

def fit_label_model(L, cardinality, epochs=500, seed=42, log_freq=50, verbose=True):
    lm = LabelModel(cardinality=cardinality, verbose=verbose)
    lm.fit(L_train=L, n_epochs=epochs, log_freq=log_freq, seed=seed)
    probs = lm.predict_proba(L)  # shape: (N, cardinality)
    return lm, probs

def hard_labels_from_probs(probs, threshold, abstain_val=ABSTAIN):
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    pred[conf < threshold] = abstain_val
    return pred, conf

# --- Subjects ---
label_model_subj, probs_subj = fit_label_model(L_subject, SUBJECT_CARD, epochs=500, seed=42, log_freq=100)
pred_subj, conf_subj = hard_labels_from_probs(probs_subj, SUBJECT_THRES)

# --- Tactics ---
label_model_tac, probs_tac = fit_label_model(L_tactics, TACTIC_CARD, epochs=500, seed=42, log_freq=100)
pred_tac, conf_tac = hard_labels_from_probs(probs_tac, TACTIC_THRES)

# --- Attach to df ---
data["subject_label"] = pred_subj
data["subject_conf"]  = conf_subj
data["tactic_label"]  = pred_tac
data["tactic_conf"]   = conf_tac

# --- Report ---
N = len(data)
subj_keep = int((data["subject_label"] != ABSTAIN).sum())
tac_keep  = int((data["tactic_label"]  != ABSTAIN).sum())

print(f"[Subjects] kept {subj_keep}/{N} rows at threshold={SUBJECT_THRES:.2f} "
      f"({subj_keep / max(N,1):.1%})")
print(f"[Tactics ] kept {tac_keep}/{N} rows at threshold={TACTIC_THRES:.2f} "
      f"({tac_keep / max(N,1):.1%})")

os.makedirs("outputs", exist_ok=True)
outpath = "outputs/weaklabels_hard.csv"
data.to_csv(outpath, index=False)
print(f"[INFO] Saved weak labels to {outpath}")

