import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/aditya/neural-proof-assistant/backend/scripts/outputs/weaklabels_hard.csv')

LABEL_COL_SUBJECTS = "subject_label"
df = df[df[LABEL_COL_SUBJECTS] != -1] # drop abstains

X = df['text'].astype(str).tolist()
y = df[LABEL_COL_SUBJECTS].astype(int).tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_vec, y_train)
y_pred = logreg.predict(X_test_vec)
print("LOGREG:")
print(" LogReg Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, logreg.predict(X_test_vec)))

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
y_pred_nb = nb.predict(X_test_vec)
print("Naive Bayes:")
print(" NB Accuracy", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20, random_state=42)
mlp.fit(X_train_vec, y_train)
y_pred_mlp = mlp.predict(X_test_vec)
print("MLP:")
print(" MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))