import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import KFold, train_test_split
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

def run(seed=0):

    n_fold = 10
    max_n_features = 64
    max_topK = 32
    
    # ------------------------ read data ------------------------
    np.random.seed(seed)
    df = pd.read_csv('icu_first24hours.csv')

    MAX_MISSING_RATE = 1.0
    df_missing_rate = df.isnull().mean().sort_values().reset_index()
    df_missing_rate.columns = ['col','missing_rate']
    cols = list(df_missing_rate[df_missing_rate['missing_rate'] < MAX_MISSING_RATE].col.values)
    df = df[cols]
    
    shuffle_idx = np.random.permutation(df.shape[0])
    split_idx = int(0.8*df.shape[0])
    train_idx = shuffle_idx[:split_idx]
    test_idx = shuffle_idx[split_idx:]
    df_test = df.iloc[test_idx]
    df = df.iloc[train_idx]
    print('train/val set: ', df.shape, '; test set: ', df_test.shape)
    
    x_cols = ['age_month', 'gender_is_male'] + cols[6:]
    X = np.nan_to_num(df[x_cols].values)
    X_test = np.nan_to_num(df_test[x_cols].values)
    y = df['HOSPITAL_EXPIRE_FLAG'].values
    y_test = df_test['HOSPITAL_EXPIRE_FLAG'].values

    # ------------------------ Rank all feats by RF ------------------------
    print('Rank all feats by RF ...')
    all_res = []
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    feature_scores = []
    for train_index, val_index in kf.split(X):
        tmp_res = []
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # RF
        m = RF(n_estimators=100, random_state=seed)
        m.fit(X_train, y_train)
        y_pred = m.predict_proba(X_val)[:,1]
        t_res = my_eval(y_val, y_pred)
        tmp_res.extend(list(t_res.values()))
        
        feature_scores.append(m.feature_importances_)
        all_res.append(tmp_res)
        
    feature_scores = np.mean(np.array(feature_scores), axis=0)
    df_imp = pd.DataFrame({'col':x_cols, 'score':feature_scores})
    df_imp = df_imp.merge(df_missing_rate, left_on='col', right_on='col', how='left')
    df_imp = df_imp.sort_values(by='score', ascending=False)
    plt.figure()
    plt.plot(df_imp.score.values)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    
    all_res = np.array(all_res)
    res_mean = np.mean(all_res, axis=0)
    res_std = np.std(all_res, axis=0)
    print('RF on all features: ', res_mean, res_std)
    
    # ------------------------ Select top feats by cross validation ------------------------
    print('Select top feats by cross validation ...')
    n_features = list(range(1,max_n_features+1))
    all_res_1 = []
    for topK in tqdm(n_features, desc='feature selection'):
        tmp_res = []
        x_cols = df_imp[:topK].col.values
        X = np.nan_to_num(df[x_cols].values)
        X_test = np.nan_to_num(df_test[x_cols].values)
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
        for train_index, val_index in kf.split(X):
            
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            m = LR()
            m.fit(X_train, y_train)
            y_pred = m.predict_proba(X_val)[:,1]
            t_res = my_eval(y_val, y_pred)
            tmp_res.append(list(t_res.values()))

        all_res_1.append(tmp_res)
    
    all_res_1 = np.array(all_res_1)
    res_mean_1 = np.mean(all_res_1, axis=1)
    res_std_1 = np.std(all_res_1, axis=1)
    res_df = pd.DataFrame(np.concatenate([np.array([n_features]).T, res_mean_1, res_std_1], axis=1))
    res_df.columns = ['topK', 'AUROC_mean', 'AUPRC_mean', 'AUROC_std', 'AUPRC_std']
    # print(res_df)
    
    plt.figure()
    plt.plot(res_df.AUROC_mean.values)
    plt.xlabel('Number of Features')
    plt.ylabel('AUROC')    
        
    # ------------------------ Build base models by cross validation ------------------------
    print('Build base models by cross validation ...')
    
    topK = np.min([max_topK, np.argmax(res_df.AUROC_mean)])
    print('topK features: {}'.format(topK))
    
    x_cols = list(df_imp[:topK].col.values)
    X = np.nan_to_num(df[x_cols].values)
    X_test = np.nan_to_num(df_test[x_cols].values)
    
    train_res = []
    val_res = []
    test_res = []
    model_df = []

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for train_index, val_index in kf.split(X):

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # LR
        m = LR(solver='liblinear', max_iter=10000)
        m.fit(X_train, y_train)
        y_pred_train = m.predict_proba(X_train)[:,1]
        y_pred_val = m.predict_proba(X_val)[:,1]
        y_pred_test = m.predict_proba(X_test)[:,1]
        t_res_train = my_eval(y_train, y_pred_train)
        t_res_val = my_eval(y_val, y_pred_val)
        t_res_test = my_eval(y_test, y_pred_test)
        
        train_res.append(list(t_res_train.values()))
        val_res.append(list(t_res_val.values()))
        test_res.append(list(t_res_test.values()))
        model_df.append(list(m.coef_[0])+list(m.intercept_))

    train_res_df = pd.DataFrame(train_res, columns=['AUROC', 'AUPRC'])
    val_res_df = pd.DataFrame(val_res, columns=['AUROC', 'AUPRC'])
    test_res_df = pd.DataFrame(test_res, columns=['AUROC', 'AUPRC'])
    model_df = pd.DataFrame(model_df, columns=x_cols+['intercept'])
    # print('{}-fold Cross validation on training set:'.format(n_fold))
    # print(train_res_df)
    print('{}-fold Cross validation on test set:'.format(n_fold))
    print(test_res_df)
    
    # ------------------------ weights ensemble for final model ------------------------
    print('Construct Final Model ...')
    final_m_coef_ = np.mean(model_df.values, axis=0)[:-1]
    final_m_intercept_ = np.mean(model_df.values, axis=0)[-1]
    model_str = ''
    for i in range(len(x_cols)):
        model_str += '{:.6f}*{} + '.format(final_m_coef_[i], x_cols[i])
    model_str += '{:.6f}'.format(final_m_intercept_)
    model_str = 'sigmoid(' + model_str + ')'
    print('Final model: Probability =', model_str)
    
    final_res = []
    # ours
    y_pred_ours = sigmoid(np.einsum('nd,d->n', X_test, final_m_coef_) + final_m_intercept_)
    final_res_ours = my_eval(y_test, y_pred_ours)
    print('Ours: ', final_res_ours)
    final_res.append([final_res_ours['auroc'], final_res_ours['auprc'], topK])
    fpr_ours, tpr_ours, _ = roc_curve(y_test, y_pred_ours)

    # baseline LR
    m = LR(solver='liblinear', max_iter=10000)
    m.fit(X, y)
    y_pred_lr = m.predict_proba(X_test)[:,1]
    final_res_lr = my_eval(y_test, y_pred_lr)
    print('lr: ', final_res_lr)
    final_res.append([final_res_lr['auroc'], final_res_lr['auprc'], topK])
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
    
#     # baseline prism_iii
#     y_pred_prism_iii = prism_iii(df_test)
#     final_res_prism_iii = my_eval(y_test, y_pred_prism_iii)
#     print('prism_iii: ', final_res_prism_iii)
#     final_res.append([final_res_prism_iii['auroc'], final_res_prism_iii['auprc'], topK])
#     fpr_prism_iii, tpr_prism_iii, _ = roc_curve(y_test, y_pred_prism_iii)
    
#     # plot
#     plt.figure()
#     lw = 2
#     plt.plot(fpr_ours, tpr_ours, color='darkorange', lw=lw, label='Ours (area = %0.6f)' % my_eval(y_test, y_pred_ours)['auroc'])
#     plt.plot(fpr_prism_iii, tpr_prism_iii, color='k', lw=lw, label='PRISM III (area = %0.6f)' % my_eval(y_test, y_pred_prism_iii)['auroc'])
#     plt.plot(fpr_lr, tpr_lr, color='b', lw=lw, label='LR (area = %0.6f)' % my_eval(y_test, y_pred_lr)['auroc'])
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()
        
    return final_res
    
    

if __name__ == "__main__":    
    
    out = []
    for i in range(100):
        print('='*60)
        print('seed:', i)
        tmp_res = run(i)
        out.append(tmp_res)
    
        res = np.array(out)
        res_mean = np.mean(res, axis=0)
        res_std = np.std(res, axis=0)
        print(res_mean)
        print(res_std)
        np.save('res/res.npy', res)
        np.save('res/res_mean.npy', res_mean)
        np.save('res/res_std.npy', res_std)
    
    
    
    