import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_curve, roc_auc_score

from baseline_prism_iii import prism_iii

def main():
    test_data=pd.read_csv('mimiciii_data/test_data.csv')

    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    logreg = LR(penalty=penalty, C=args.C, random_state=42)
    logreg.fit(train_X, train_y)