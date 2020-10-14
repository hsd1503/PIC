"""
Shenda, Oct 13, 2020

- add cross-validation
- remove random forest final model
- change first24hours to first48hours
- chnage MAX_MISSING_RATE from 1.0 to 0.4 (same as mimic-iii exp)

"""

import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import KFold

from baseline_prism_iii import prism_iii

import warnings
warnings.filterwarnings('ignore') 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

plt.rcParams['pdf.fonttype'] = 42

def my_eval(gt, y_pred_proba):
    """
    y_pred_proba are float
    gt, y_pred are binary
    """
    
    y_pred = np.array(y_pred_proba > 0.5, dtype=int)

    ret = OrderedDict({})
    ret['auroc'] = roc_auc_score(gt, y_pred_proba)
    ret['auprc'] = average_precision_score(gt, y_pred_proba)
    ret['precision'] = precision_score(gt, y_pred)
    ret['recall'] = recall_score(gt, y_pred)
    ret['accuracy'] = accuracy_score(gt, y_pred)
    ret['f1'] = f1_score(gt, y_pred)

    return list(ret.items())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def print_lr_model(m, x_cols):
    
    final_m_coef_ = m.coef_[0]
    final_m_intercept_ = m.intercept_[0]
    model_str = ''
    for i in range(len(x_cols)):
        model_str += '{:.4f}*{} + '.format(final_m_coef_[i], x_cols[i])
    model_str += '{:.4f}'.format(final_m_intercept_)
    model_str = 'sigmoid(' + model_str + ')'
    print('Final model: Probability =', model_str)
    
    
if __name__ == "__main__":    
    
    seed = 0
    n_fold = 10
    n_bootstrap = 10
    max_n_features = 16
    
    ### read data
    np.random.seed(seed)
    df = pd.read_csv('icu_first48hours.csv')
    
    ### filter by missing rate
    MAX_MISSING_RATE = 1.0 # use all features
    df_missing_rate = df.isnull().mean().sort_values().reset_index()
    df_missing_rate.columns = ['col','missing_rate']
    cols = list(df_missing_rate[df_missing_rate['missing_rate'] < MAX_MISSING_RATE].col.values)
    df = df[cols]

    x_cols_all = ['age_month', 'gender_is_male'] + cols[6:]
    df_X = df[x_cols_all]
    X = np.nan_to_num(df_X.values)
    y = df['HOSPITAL_EXPIRE_FLAG'].values

    ### loop
    feature_ranks_all = []
    kf_outer = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for dev_index, test_index in tqdm(kf_outer.split(X), desc='feature ranking outer'):
        X_dev, X_test = X[dev_index], X[test_index]
        y_dev, y_test = y[dev_index], y[test_index]

        kf_inner = KFold(n_splits=n_bootstrap, shuffle=True, random_state=seed)
        for train_index, val_index in tqdm(kf_inner.split(X_dev), desc='feature ranking inner'):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
        
            # use RF to rank features
            m = RF(n_estimators=100, random_state=seed)
            m.fit(X_train, y_train)        
            feature_ranks_all.append(m.feature_importances_.argsort()[::-1])

    feature_ranks_all = np.array(feature_ranks_all)
    feature_ranks = np.mean(feature_ranks_all, axis=0, dtype=int).argsort()
        
    # Evaluation
    all_res = []
    kf_outer = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for dev_index, test_index in kf_outer.split(X):
        df_test = df_X.iloc[test_index]
    
        tmp_res = []
        for topK in tqdm(range(1, max_n_features+1), desc='evaluation'):
            X_dev, X_test = X[dev_index], X[test_index]
            y_dev, y_test = y[dev_index], y[test_index]
            tmp_X_dev = X_dev[:, feature_ranks[:topK]]
            tmp_X_test = X_test[:, feature_ranks[:topK]]

            m = LR()
            m.fit(tmp_X_dev, y_dev)
            y_pred = m.predict_proba(tmp_X_test)[:,1]
            t_res = my_eval(y_test, y_pred)
            tmp_res.append(t_res)
        
        y_pred_prism_iii = prism_iii(df_test)
        tmp_res.append(my_eval(y_test, y_pred_prism_iii))

        all_res.append(tmp_res)

    all_res = np.array(all_res)
    print(all_res.shape)

    run_id = 'first'
    out = {'all_res':all_res}
    with open('res1/{}.pkl'.format(run_id), 'wb') as fout:
        pickle.dump(out, fout)