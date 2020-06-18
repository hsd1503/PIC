import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix

from baseline_prism_iii import prism_iii
import warnings
warnings.filterwarnings('ignore') 
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

    # ------------------------ Cross Validation ------------------------
    print('Cross validation ...')
    
    x_cols = cols['all']
    X = np.nan_to_num(df[x_cols].values)
    y = df['HOSPITAL_EXPIRE_FLAG'].values
    
    train_res = []
    test_res = []
    model_df = []

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # LR
        m = LR(solver='liblinear', max_iter=10000)
        m.fit(X_train, y_train)
        y_pred_train = m.predict_proba(X_train)[:,1]
        y_pred_test = m.predict_proba(X_test)[:,1]
        t_res_train = my_eval(y_train, y_pred_train)
        t_res_test = my_eval(y_test, y_pred_test)
        
        train_res.append(list(t_res_train.values()))
        test_res.append(list(t_res_test.values()))
        model_df.append(list(m.coef_[0])+list(m.intercept_))

    train_res_df = pd.DataFrame(train_res, columns=['AUROC', 'AUPRC'])
    test_res_df = pd.DataFrame(test_res, columns=['AUROC', 'AUPRC'])
    model_df = pd.DataFrame(model_df, columns=x_cols+['intercept'])
    print('5-fold Cross validation on training set:')
    print(train_res_df)
    print('5-fold Cross validation on test set:')
    print(test_res_df)
    
    # ------------------------ final model ------------------------
    print('Construct Final Model ...')
    final_m_coef_ = np.mean(model_df.values, axis=0)[:-1]
    final_m_intercept_ = np.mean(model_df.values, axis=0)[-1]
    model_str = ''
    for i in range(len(cols_all)):
        model_str += '{:.6f}*{} + '.format(final_m_coef_[i], cols_all[i])
    model_str += '{:.6f}'.format(final_m_intercept_)
    model_str = 'sigmoid(' + model_str + ')'
    print('Final model: Probability =', model_str)
    
    y_pred_ours = sigmoid(np.einsum('nd,d->n', X, final_m_coef_) + final_m_intercept_)
    print('Ours: ', my_eval(y, y_pred_ours))
    fpr_ours, tpr_ours, _ = roc_curve(y, y_pred_ours)
        
    y_pred_prism_iii = prism_iii(df)
    print('prism_iii: ', my_eval(y, y_pred_prism_iii))
    fpr_prism_iii, tpr_prism_iii, _ = roc_curve(y, y_pred_prism_iii)
    
    plt.figure()
    lw = 2
    plt.plot(fpr_ours, tpr_ours, color='darkorange', lw=lw, label='Ours (area = %0.4f)' % my_eval(y, y_pred_ours)['auroc'])
    plt.plot(fpr_prism_iii, tpr_prism_iii, color='k', lw=lw, label='PRISM III (area = %0.4f)' % my_eval(y, y_pred_prism_iii)['auroc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()    
    