import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

from baseline_prism_iii import prism_iii
from util import my_eval

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

"""
PRISM_III OrderedDict([('auroc', 0.5949378836721735), ('auprc', 0.15335782217228863)])
all [0.70997    0.17301496 0.82774849 0.29295686] [0.01570708 0.00756412 0.01419626 0.00952535]

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
    
    # ------------------------ prism_iii ------------------------
    y_pred = prism_iii(df)
    print('>>>>>>>>>>', 'PRISM_III', my_eval(y, y_pred))

    # ------------------------ all feats ------------------------
    all_res = []
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    feature_scores = []
    for train_index, test_index in kf.split(X):
        tmp_res = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train.shape, X_test.shape)
        print(Counter(y_train), Counter(y_test))
        
        # LR
        m = LR()
        m.fit(X_train, y_train)
        y_pred = m.predict_proba(X_test)[:,1]
        t_res = my_eval(y_test, y_pred)
        print('LR', t_res)
        tmp_res.extend(list(t_res.values()))
        
        # RF
        m = RF(n_estimators=100, random_state=seed)
        m.fit(X_train, y_train)
        y_pred = m.predict_proba(X_test)[:,1]
        t_res = my_eval(y_test, y_pred)
        print('RF', t_res)
        tmp_res.extend(list(t_res.values()))
        
        feature_scores.append(m.feature_importances_)
        all_res.append(tmp_res)
        
    feature_scores = np.mean(np.array(feature_scores), axis=0)
    df_imp = pd.DataFrame({'col':x_cols, 'score':feature_scores})
    df_imp = df_imp.merge(df_missing_rate, left_on='col', right_on='col', how='left')
    df_imp = df_imp.sort_values(by='score', ascending=False)
    plt.plot(df_imp.score.values)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    df_imp.to_csv('res/features.csv', index=False)
    
    all_res = np.array(all_res)
    res_mean = np.mean(all_res, axis=0)
    res_std = np.std(all_res, axis=0)
    print('>>>>>>>>>> all \n', res_mean, res_std)
    
