#%%
import os
from os import path
import sys


# deep learning
import numpy as np

# data 
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

bt500 = pd.read_csv('Test_2_iqa.csv')

subj = 'BT-500'
met = 'Topiq-fr'
subj_col = bt500[subj]
met_col = bt500[met]

PLCC, SRCC, KTCC = pearsonr(subj_col,met_col)[0], spearmanr(subj_col,met_col)[0], kendalltau(subj_col,met_col)[0]
print("PLCC: ", PLCC)
print("SRCC: ", SRCC)
print("KTCC: ", KTCC)
print(len(subj_col.values))
# %%
