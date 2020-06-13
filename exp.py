import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

"""
seed=0
Counter({0: 12478, 1: 971})
(10759, 1185) (2690, 1185)
Counter({0: 9975, 1: 784}) Counter({0: 2503, 1: 187})
OrderedDict([('auroc', 0.8609732492132436), ('auprc', 0.3461800277643663)])
(10759, 1185) (2690, 1185)
Counter({0: 9986, 1: 773}) Counter({0: 2492, 1: 198})
OrderedDict([('auroc', 0.8701876307213385), ('auprc', 0.35290723542182667)])
(10759, 1185) (2690, 1185)
Counter({0: 9983, 1: 776}) Counter({0: 2495, 1: 195})
OrderedDict([('auroc', 0.8660058578695854), ('auprc', 0.3450080756540379)])
(10759, 1185) (2690, 1185)
Counter({0: 9981, 1: 778}) Counter({0: 2497, 1: 193})
OrderedDict([('auroc', 0.8679493111941584), ('auprc', 0.34499688203714823)])
(10760, 1185) (2689, 1185)
Counter({0: 9987, 1: 773}) Counter({0: 2491, 1: 198})
OrderedDict([('auroc', 0.8354895806722382), ('auprc', 0.33612982499000876)])
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
    df = pd.read_csv('icu_first48hours.csv')

    MAX_MISSING_RATE = 1.0
    df_missing_rate = df.isnull().mean().sort_values().reset_index()
    df_missing_rate.columns = ['col','missing_rate']

    print(df_missing_rate[df_missing_rate['missing_rate'] < MAX_MISSING_RATE].shape)


    cols = list(df_missing_rate[df_missing_rate['missing_rate'] < MAX_MISSING_RATE].col.values)
    final_df = df[cols]
    x_cols = ['age_month', 'gender_is_male'] + cols[6:]

    X = final_df[x_cols].values
    X = np.nan_to_num(X)
    y = final_df['HOSPITAL_EXPIRE_FLAG'].values
    print(Counter(y))

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    feature_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train.shape, X_test.shape)
        print(Counter(y_train), Counter(y_test))
        
        m = RF(n_estimators=100, random_state=seed)
        m.fit(X_train, y_train)
        y_pred = m.predict_proba(X_test)[:,1]
        print(my_eval(y_test, y_pred))
        
        feature_scores.append(m.feature_importances_)
        
    feature_scores = np.mean(np.array(feature_scores), axis=0)
    df_imp = pd.DataFrame({'col':x_cols, 'score':feature_scores})
    df_imp = df_imp.sort_values(by='score', ascending=False)
    plt.plot(df_imp.score.values[1:])
    
    
    
    
    
    