import pandas as pd
from collections import Counter, OrderedDict
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

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


    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train.shape, X_test.shape)
        print(Counter(y_train), Counter(y_test))
        
        m = RF(n_estimators=1000)
        m.fit(X_train, y_train)
        y_pred = m.predict_proba(X_test)[:,1]
        print(my_eval(y_test, y_pred))
        