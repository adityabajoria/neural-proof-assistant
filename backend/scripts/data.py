# pip install datasets pandas
import pandas as pd
import numpy as np
from datasets import load_dataset

OUT_FILE = "unlabeled_10k_combined.csv"

# --- 1) Load & sample 5k from your synthetic data ---
syn = pd.read_csv("unlabeled_goals.csv")
syn_5k = syn.sample(n=5000, random_state=42).reset_index(drop=True)
syn_5k["source"] = "synthetic"

# --- 2) Load & sample 5k from Proof-Pile-2 (arXiv math subset) ---
ds = load_dataset("EleutherAI/proof-pile-2", "arxiv")["train"]
idx = np.random.RandomState(42).choice(len(ds), size=5000, replace=False)
texts = [ds[i]["text"] for i in idx]

real_5k = pd.DataFrame({"text": texts})
real_5k["source"] = "proof-pile-2_arxiv"
real_5k["text"] = (real_5k["text"].astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip())
real_5k = real_5k[real_5k["text"].str.len().between(30, 1200)].reset_index(drop=True)

# --- 3) Merge into one dataset ---
combined = pd.concat([syn_5k, real_5k], ignore_index=True)
combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

# --- 4) Shuffle rows for randomness ---
combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

# --- 5) Save final file ---
combined.to_csv(OUT_FILE, index=False)
print(f"[INFO] Saved {OUT_FILE} with {len(combined)} rows")
print(combined["source"].value_counts())