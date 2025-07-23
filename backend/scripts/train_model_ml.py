import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

proofnet = pd.read_csv('/Users/aditya/neural-proof-assistant/backend/data/proofnet/proofnet.csv')
useful_cols = ['nl_statement', 'first_step', 'tactic']
proofnet = proofnet[useful_cols]
proofnet['combined_text'] = proofnet['nl_statement'] + " " + proofnet['first_step']
print(proofnet.head())
