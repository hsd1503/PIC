import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

from baseline_prism_iii import prism_iii
from util import my_eval

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

"""
best K = 29

128 [0.76124024 0.19582179 0.83073421 0.29864357] [0.01315973 0.02189819 0.01556883 0.01308301]
64 [0.7805817  0.22081772 0.82316914 0.29943084] [0.01680027 0.02204565 0.00761315 0.00708166]
32 [0.80543884 0.32258149 0.81460239 0.30448361] [0.00901511 0.01479886 0.01769794 0.01152791]
16 [0.79642514 0.31292076 0.77292331 0.27221166] [0.01429078 0.01415364 0.02113059 0.01094292]
8 [0.68320853 0.26590942 0.66487007 0.240117  ] [0.0078789  0.01751636 0.01982672 0.01322394]
4 [0.65517163 0.24284718 0.64235016 0.20290579] [0.01284803 0.02097817 0.01920278 0.02079366]

"""

if __name__ == "__main__":

    seed = 0
    df = pd.read_csv('icu_first24hours.csv')

    MAX_MISSING_RATE = 1.0
    df_missing_rate = df.isnull().mean().sort_values().reset_index()
    df_missing_rate.columns = ['col','missing_rate']
    cols = list(df_missing_rate[df_missing_rate['missing_rate'] < MAX_MISSING_RATE].col.values)
    final_df = df[cols]
    x_cols = ['age_month', 'gender_is_male'] + cols[6:]

    X = final_df[x_cols].values
    X = np.nan_to_num(X) # impute zero
    y = final_df['HOSPITAL_EXPIRE_FLAG'].values
    print(Counter(y))
    
    df_imp = pd.read_csv('res/features.csv')
    
    # ------------------------ top feats ------------------------
    n_features = list(range(1,129))
    all_res_1 = []
    for topK in tqdm(n_features):
        tmp_res = []
        x_cols = df_imp[:topK].col.values
        X = df[x_cols].values
        X = np.nan_to_num(X) # impute zero
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(X):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # LR
            m = LR()
            m.fit(X_train, y_train)
            y_pred = m.predict_proba(X_test)[:,1]
            t_res = my_eval(y_test, y_pred)
            tmp_res.append(list(t_res.values()))

        all_res_1.append(tmp_res)
    
    all_res_1 = np.array(all_res_1)
    res_mean_1 = np.mean(all_res_1, axis=1)
    res_std_1 = np.std(all_res_1, axis=1)
    res_df = pd.DataFrame(np.concatenate([np.array([n_features]).T, res_mean_1, res_std_1], axis=1))
    res_df.columns = ['topK', 'AUROC_mean', 'AUPRC_mean', 'AUROC_std', 'AUPRC_std']
    print(res_df)
    res_df.to_csv('res/features_perf.csv', index=False)