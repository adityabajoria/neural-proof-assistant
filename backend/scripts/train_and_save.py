import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = '/Users/aditya/neural-proof-assistant/backend/data/outputs/weaklabels_hard.csv'
LABEL_COL_SUBJ = 'subject_label'
LABEL_COL_TAC = 'tactic_label'
OUTPUT_DIR = "."
SUBJ_DIR = os.path.join(OUTPUT_DIR, "models", "subject")
TAC_DIR = os.path.join(OUTPUT_DIR, "models", "tactic")
os.makedirs(SUBJ_DIR, exist_ok=True)
os.makedirs(TAC_DIR, exist_ok=True)


# train-test split
df = pd.read_csv(CSV_PATH)

subject = df[df[LABEL_COL_SUBJ] != -1]
X_text_subj = subject['text'].astype(str).tolist()
y_raw_subj = subject[LABEL_COL_SUBJ].astype(int).astype(str).tolist()
le_subj = LabelEncoder()
y_subj = le_subj.fit_transform(y_raw_subj)

tactic = df[df[LABEL_COL_TAC] != -1]
X_text_tac = tactic['text'].astype(str).tolist()
y_raw_tac = tactic[LABEL_COL_TAC].astype(int).astype(str).tolist()
le_tac = LabelEncoder()
y_tac = le_tac.fit_transform(y_raw_tac)

# Train Test Split (Subject)
X_train_txt_subj, X_test_txt_subj, y_train, y_test = train_test_split(X_text_subj, y_subj, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_subj = vectorizer.fit_transform(X_train_txt_subj)
X_test_subj = vectorizer.transform(X_test_txt_subj)

# Train Test Split (Tactic)
X_train_txt_tac, X_test_txt_tac, y_train_tac, y_test_tac = train_test_split(X_text_tac, y_tac, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tac = vectorizer.fit_transform(X_train_txt_tac)
X_test_tac = vectorizer.transform(X_test_txt_tac)

def train_models(X_train, y_train):
    models = {}

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    models['logreg'] = logreg

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models['nb'] = nb

    # MLP
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    models['mlp'] = mlp

    return models

models_subj = train_models(X_train_subj, y_train)
models_tac = train_models(X_train_tac, y_train_tac)

print("SUBJECT MODELS")
for name, m in models_subj.items():
    y_pred = m.predict(X_test_subj)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[{name}] Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

print("TACTIC MODELS")
for name, m in models_tac.items():
    y_pred_tac = m.predict(X_test_tac)
    acc = accuracy_score(y_test_tac, y_pred_tac)
    print(f"\n[{name}] Accuracy: {acc:.4f}")
    print(classification_report(y_test_tac, y_pred_tac, digits=3))

joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "vectorizer.pkl"))
joblib.dump(le_subj, os.path.join(OUTPUT_DIR, "label_encoder_subject.pkl"))
joblib.dump(le_tac, os.path.join(OUTPUT_DIR, "label_encoder_tactic.pkl"))

for name, m in models_subj.items() and models_tac.items():
    joblib.dump(m, os.path.join(SUBJ_DIR, f"{name}.pkl"))
    joblib.dump(m, os.path.join(TAC_DIR, f"{name}.pkl"))

print("\n[INFO] Saved:")
print(" - vectorizer.pkl")
print(" - label_encoder_subject.pkl")
print(f" - models in {SUBJ_DIR} (logreg.pkl, nb.pkl, mlp.pkl)")
print(f" - models in {TAC_DIR} (logreg.pkl, nb.pkl, mlp.pkl)")