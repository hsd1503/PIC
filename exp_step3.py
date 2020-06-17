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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def my_eval(gt, y_pred_proba):
    """
    y_pred_proba are float
    gt, y_pred are binary
    """
    
    ret = OrderedDict({})
    ret['auroc'] = roc_auc_score(gt, y_pred_proba)
    ret['auprc'] = average_precision_score(gt, y_pred_proba)

    return ret

if __name__ == "__main__":

    seed = 0
    df = pd.read_csv('icu_first24hours.csv')

    cols = {}
    # 人口学资料
    cols['demographics'] = ['age_month']
    # 生命体征
    cols['vitals'] = ['chart_1001_max','chart_1004_max','chart_1004_min','chart_1015_min','chart_1016_max','chart_1016_min']
    # 外周血象CBC
    cols['cbc'] = ['lab_5129_max','lab_5252_min']
    # 凝血功能
    cols['pt'] = ['lab_5174_min','lab_5186_max']
    # 血气分析
    cols['gas'] = ['lab_5211_max','lab_5211_min','lab_5224_min','lab_5227_max','lab_5233_max','lab_5235_max','lab_5237_min','lab_5248_min','lab_5249_max','lab_5249_min','lab_5226_max']
    # 血清电解质
    cols['chemistry'] = ['lab_5215_max','lab_5218_min']
    # 血糖
    cols['glucose'] = ['lab_5223_max']
    # 全部
    cols_all = []
    for k, v in cols.items():
        cols_all += v
    cols['all'] = cols_all

    models = ['vitals', 'cbc', 'pt', 'gas', 'chemistry', 'glucose', 'all']

    # ------------------------ all feats ------------------------
    all_res = []

    for model in models:
        tmp_res = []

        x_cols = cols['demographics'] + cols[model]
        X = np.nan_to_num(df[x_cols].values)
        y = df['HOSPITAL_EXPIRE_FLAG'].values

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
            
        all_res.append(tmp_res)

    
    all_res = np.array(all_res)
    res_mean = np.mean(all_res, axis=1)
    res_std = np.std(all_res, axis=1)
    print('>>>>>>>>>> all \n', res_mean, res_std)
