import pandas as pd
import re

proofnet = pd.read_csv('/Users/aditya/neural-proof-assistant/backend/data/proofnet/proofnet.csv')
useful_cols = ['nl_statement', 'first_step', 'tactic']
proofnet = proofnet[useful_cols]
print(proofnet.head())
