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
# pd.set_option('display.max_rows', None)

"""
PRISM_III OrderedDict([('auroc', 0.5949378836721735), ('auprc', 0.15335782217228863)])

all [0.70997    0.17301496 0.82774849 0.29295686] [0.01570708 0.00756412 0.01419626 0.00952535]

128 [0.76124024 0.19582179 0.83073421 0.29864357] [0.01315973 0.02189819 0.01556883 0.01308301]
64 [0.7805817  0.22081772 0.82316914 0.29943084] [0.01680027 0.02204565 0.00761315 0.00708166]
32 [0.80543884 0.32258149 0.81460239 0.30448361] [0.00901511 0.01479886 0.01769794 0.01152791]
16 [0.79642514 0.31292076 0.77292331 0.27221166] [0.01429078 0.01415364 0.02113059 0.01094292]
8 [0.68320853 0.26590942 0.66487007 0.240117  ] [0.0078789  0.01751636 0.01982672 0.01322394]
4 [0.65517163 0.24284718 0.64235016 0.20290579] [0.01284803 0.02097817 0.01920278 0.02079366]


128,5211,实际碱剩余 ABE,Base Excess,Blood,Blood Gas,11555-0
140,5227,乳酸 Lac,Lactate,Blood,Blood Gas,32693-4
160,5249,标准碱剩余 SBE,Base Excess,Blood,Blood Gas,11555-0
147,5235,二氧化碳分压 pCO2,pCO2,Blood,Blood Gas,11557-6
149,5237,酸碱度 pH,pH,Blood,Blood Gas,11558-4
159,5248,标准碳酸氢根 SBC,Bicarbonate,Blood,Chemistry,23224

"""

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

    MAX_MISSING_RATE = 1.0
    df_missing_rate = df.isnull().mean().sort_values().reset_index()
    df_missing_rate.columns = ['col','missing_rate']

    print(df_missing_rate[df_missing_rate['missing_rate'] < MAX_MISSING_RATE].shape)

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
    
    all_res = np.array(all_res)
    res_mean = np.mean(all_res, axis=0)
    res_std = np.std(all_res, axis=0)
    print('>>>>>>>>>> all \n', res_mean, res_std)
    
    # ------------------------ top feats ------------------------
    for topK in [128, 64, 32, 16, 8, 4]:
        all_res_1 = []
        x_cols = df_imp[:topK].col.values
        X = final_df[x_cols].values
        X = np.nan_to_num(X) # impute zero
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
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

            all_res_1.append(tmp_res)
    
        all_res_1 = np.array(all_res_1)
        res_mean_1 = np.mean(all_res_1, axis=0)
        res_std_1 = np.std(all_res_1, axis=0)
        print('>>>>>>>>>> {} \n'.format(topK), res_mean_1, res_std_1)
    
    