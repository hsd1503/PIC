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
        
        # RF
        m = RF(n_estimators=100, random_state=seed)
        m.fit(X_train, y_train)
        y_pred = m.predict_proba(X_test)[:,1]
        t_res = my_eval(y_test, y_pred)
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
    n_features = list(range(1,65))
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
    
    plt.plot(res_df.AUROC_mean.values)
    plt.xlabel('Number of Features')
    plt.ylabel('AUROC')    
        
    # ------------------------ Cross Validation ------------------------
    print('Cross validation ...')
    
    topK = np.min(32, np.argmax(res_df.AUROC_mean))
    print('topK: {}'.format(topK))
    
    x_cols = df_imp[:topK].col.values
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
    plt.plot(fpr_ours, tpr_ours, color='darkorange', lw=lw, label='Ours (area = %0.6f)' % my_eval(y, y_pred_ours)['auroc'])
    plt.plot(fpr_prism_iii, tpr_prism_iii, color='k', lw=lw, label='PRISM III (area = %0.6f)' % my_eval(y, y_pred_prism_iii)['auroc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()    
        
    
    
    
    
    
    
    
    
    
    