import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.io import arff

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from cuml.svm import SVC
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression


def run_svm(scaled_df):
    
    clf = SVC()
    C_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    gamma_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 'auto']

    search_params = {
        'C': C_list,
        'gamma': gamma_list
    }
    
    metrics = ['accuracy', 'f1', 'roc_auc']

    raw_train_arr = []
    raw_test_arr = []
    
    for i in range(5):
        
        train_df = scaled_df.sample(5000, replace = True)

        test_df = scaled_df[~scaled_df.index.isin(train_df.index)]
        
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.y
        
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.y
        
        search_results = GridSearchCV(clf, search_params, scoring = metrics, refit = False)
        
        search_results.fit(X_train, y_train)
        
        results = pd.DataFrame(search_results.cv_results_['params'])

        results['mean_accuracy'] = search_results.cv_results_['mean_test_accuracy']
        results['mean_f1'] = search_results.cv_results_['mean_test_f1']
        results['mean_auc'] = search_results.cv_results_['mean_test_roc_auc']
        
        opt_acc_inf = results.sort_values(by = 'mean_accuracy', ascending = False).iloc[0]
        opt_f1_inf = results.sort_values(by = 'mean_f1', ascending = False).iloc[0]
        opt_auc_inf = results.sort_values(by = 'mean_auc', ascending = False).iloc[0]
        
        opt_acc_clf = SVC(C = opt_acc_inf.C, gamma = opt_acc_inf.gamma)
        opt_f1_clf = SVC(C = opt_f1_inf.C, gamma = opt_f1_inf.gamma)
        opt_auc_clf = SVC(C = opt_auc_inf.C, gamma = opt_auc_inf.gamma)
        
        opt_acc_clf.fit(X_train, y_train)
        opt_f1_clf.fit(X_train, y_train)
        opt_auc_clf.fit(X_train, y_train)
        
        train_score_acc = opt_acc_clf.score(X_train, y_train)
        train_score_f1 = f1_score(y_train, opt_f1_clf.predict(X_train))
        train_score_auc = roc_auc_score(y_train, opt_auc_clf.predict(X_train))
                        
        test_score_acc = opt_acc_clf.score(X_test, y_test)
        test_score_f1 = f1_score(y_test, opt_f1_clf.predict(X_test))
        test_score_auc = roc_auc_score(y_test, opt_auc_clf.predict(X_test))
        
        raw_train_arr.append([train_score_acc, train_score_f1, train_score_auc])
        raw_test_arr.append([test_score_acc, test_score_f1, test_score_auc])
        
                
    
    raw_train_arr = np.array(raw_train_arr).reshape(5, 3)
    raw_test_arr = np.array(raw_test_arr).reshape(5, 3)
    
    raw_train_df = pd.DataFrame(data = raw_train_arr, columns = ['accuracy', 'f1', 'auc'])
    raw_test_df = pd.DataFrame(data = raw_test_arr, columns = ['accuracy', 'f1', 'auc'])
    
    return raw_train_df, raw_test_df

def GridSearch_random_forest(X_train, y_train):
    X_train = X_train.to_numpy().astype('float32')
    y_train = y_train.to_numpy().astype('float32')
    
    folds = KFold(n_splits = 5)
    
    n_estimators_lst = [128, 256, 512, 1024]
    #max_features_lst = [1, 2, 4, 6, 8, 12, 16, 20]
    #max_features_lst = [1 / item for item in max_features_lst]
    max_features_lst = ['auto', 'sqrt', 'log2']
        
    fin_arr = []    
    
    for n_estimators in n_estimators_lst:
        
        for max_features in max_features_lst:

            clf = RandomForestClassifier(n_estimators = n_estimators, max_features = max_features)

            predicted_y = []
            true_y = []

            for train, holdout in folds.split(X_train):
                clf.fit(X_train[train], y_train[train])

                predicted_y.append(clf.predict(X_train[holdout]))

                true_y.append(y_train[holdout])

            predicted_y = np.concatenate(predicted_y)
            true_y = np.concatenate(true_y)
            
            accuracy_train = accuracy_score(true_y, predicted_y)
            f1_train = f1_score(true_y, predicted_y)
            roc_auc_train = roc_auc_score(true_y, predicted_y)
            
            fin_arr.append([n_estimators, max_features, accuracy_train, f1_train, roc_auc_train])
        
    fin_arr = np.array(fin_arr).reshape((len(n_estimators_lst) * len(max_features_lst)), 5)
    
    columns = ['n_estimators', 'max_features', 'mean_accuracy', 'mean_f1', 'mean_auc']
    
    results = pd.DataFrame(data = fin_arr, columns = columns)
    results.n_estimators = results.n_estimators.astype(int)
    
    

    return results

def run_random_forest(scaled_df):
    raw_train_arr = []
    raw_test_arr = []
    
    for i in range(5):
        
        train_df = scaled_df.sample(5000, replace = True)

        test_df = scaled_df[~scaled_df.index.isin(train_df.index)]
        
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.y
        
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.y
        
        search_results = GridSearch_random_forest(X_train, y_train)
                
        results = search_results
        
        opt_acc_inf = results.sort_values(by = 'mean_accuracy', ascending = False).iloc[0]
        opt_f1_inf = results.sort_values(by = 'mean_f1', ascending = False).iloc[0]
        opt_auc_inf = results.sort_values(by = 'mean_auc', ascending = False).iloc[0]
                
        opt_acc_clf = RandomForestClassifier(n_estimators = opt_acc_inf.n_estimators,
                                             max_features = opt_acc_inf.max_features)

        opt_f1_clf = RandomForestClassifier(n_estimators = opt_f1_inf.n_estimators,
                                             max_features = opt_f1_inf.max_features)

        opt_auc_clf = RandomForestClassifier(n_estimators = opt_auc_inf.n_estimators,
                                             max_features = opt_auc_inf.max_features)
        
        X_train_np = X_train.to_numpy().astype('float32')
        y_train_np = y_train.to_numpy().astype('float32')
        
        X_test_np = X_test.to_numpy().astype('float32')
        y_test_np = y_test.to_numpy().astype('float32')
                
        opt_acc_clf.fit(X_train_np, y_train_np)
        opt_f1_clf.fit(X_train_np, y_train_np)
        opt_auc_clf.fit(X_train_np, y_train_np)
        
        train_score_acc = opt_acc_clf.score(X_train_np, y_train_np)
        train_score_f1 = f1_score(y_train_np, opt_f1_clf.predict(X_train_np))
        train_score_auc = roc_auc_score(y_train_np, opt_auc_clf.predict(X_train_np))
                        
        test_score_acc = opt_acc_clf.score(X_test_np, y_test_np)
        test_score_f1 = f1_score(y_test_np, opt_f1_clf.predict(X_test_np))
        test_score_auc = roc_auc_score(y_test_np, opt_auc_clf.predict(X_test_np))
        
        raw_train_arr.append([train_score_acc, train_score_f1, train_score_auc])
        raw_test_arr.append([test_score_acc, test_score_f1, test_score_auc])
        
                
    
    raw_train_arr = np.array(raw_train_arr).reshape(5, 3)
    raw_test_arr = np.array(raw_test_arr).reshape(5, 3)
    
    raw_train_df = pd.DataFrame(data = raw_train_arr, columns = ['accuracy', 'f1', 'auc'])
    raw_test_df = pd.DataFrame(data = raw_test_arr, columns = ['accuracy', 'f1', 'auc'])
    
    return raw_train_df, raw_test_df

def run_log_reg(scaled_df):
    
    raw_train_arr = []
    raw_test_arr = []
    
    metrics = ['accuracy', 'f1', 'roc_auc_ovo']
    
    C_vals = range(-8, 5)
    C_vals = [10 ** val for val in C_vals]
    
    penalty = ['none', 'l1', 'l2']

    params = {'penalty' : penalty, 'C' : C_vals}


    
    for i in range(5):
        
        train_df = scaled_df.sample(5000, replace = True)

        test_df = scaled_df[~scaled_df.index.isin(train_df.index)]

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.y

        X_test = test_df.iloc[:, :-1]
        y_test = test_df.y


        #classifier_none = LogisticRegression(penalty = 'none', max_iter = 100000)
        #search_results_none = GridSearchCV(classifier_none, param_grid = {}, scoring = metrics, refit = False)

        #search_results_none.fit(X_train, y_train)

        #no_penalty_df = pd.DataFrame().assign(C = [0], 
        #                                      penalty = ['none'],
        #                                      mean_accuracy = search_results_none.cv_results_['mean_test_accuracy'],
        #                                      mean_f1 = search_results_none.cv_results_['mean_test_f1'],
        #                                      mean_auc = search_results_none.cv_results_['mean_test_roc_auc_ovo']
         #                                    )
        clf = LogisticRegression()
        search_results = GridSearchCV(clf, params, scoring = metrics, refit = False)
        search_results.fit(X_train, y_train)

        results = pd.DataFrame(search_results.cv_results_['params'])

        results['mean_accuracy'] = search_results.cv_results_['mean_test_accuracy']
        results['mean_f1'] = search_results.cv_results_['mean_test_f1']
        results['mean_auc'] = search_results.cv_results_['mean_test_roc_auc_ovo']

        #results = results.append(no_penalty_df)
                
        opt_acc_inf = results.sort_values(by = 'mean_accuracy', ascending = False).iloc[0]
        opt_f1_inf = results.sort_values(by = 'mean_f1', ascending = False).iloc[0]
        opt_auc_inf = results.sort_values(by = 'mean_auc', ascending = False).iloc[0]
        
        opt_acc_clf = LogisticRegression(C = opt_acc_inf.C, penalty = opt_acc_inf.penalty, max_iter = 100000)
        opt_f1_clf = LogisticRegression(C = opt_f1_inf.C, penalty = opt_f1_inf.penalty, max_iter = 100000)
        opt_auc_clf = LogisticRegression(C = opt_auc_inf.C, penalty = opt_auc_inf.penalty, max_iter = 100000)

        
        opt_acc_clf.fit(X_train, y_train)
        opt_f1_clf.fit(X_train, y_train)
        opt_auc_clf.fit(X_train, y_train)
        
        train_score_acc = opt_acc_clf.score(X_train, y_train)
        train_score_f1 = f1_score(y_train, opt_f1_clf.predict(X_train))
        train_score_auc = roc_auc_score(y_train, opt_auc_clf.predict(X_train))
                        
        test_score_acc = opt_acc_clf.score(X_test, y_test)
        test_score_f1 = f1_score(y_test, opt_f1_clf.predict(X_test))
        test_score_auc = roc_auc_score(y_test, opt_auc_clf.predict(X_test))
        
        raw_train_arr.append([train_score_acc, train_score_f1, train_score_auc])
        raw_test_arr.append([test_score_acc, test_score_f1, test_score_auc])
        
    raw_train_arr = np.array(raw_train_arr).reshape(5, 3)
    raw_test_arr = np.array(raw_test_arr).reshape(5, 3)
    
    raw_train_df = pd.DataFrame(data = raw_train_arr, columns = ['accuracy', 'f1', 'auc'])
    raw_test_df = pd.DataFrame(data = raw_test_arr, columns = ['accuracy', 'f1', 'auc'])


    
    return raw_train_df, raw_test_df

def run_log_reg_skl(scaled_df):
    
    raw_train_arr = []
    raw_test_arr = []
    
    metrics = ['accuracy', 'f1', 'roc_auc_ovo']
    
    C_vals = range(-8, 5)
    C_vals = [10 ** val for val in C_vals]
    
    penalty = ['l1', 'l2']

    params = {'penalty' : penalty, 'C' : C_vals}


    
    for i in range(5):
        
        train_df = scaled_df.sample(5000, replace = True)

        test_df = scaled_df[~scaled_df.index.isin(train_df.index)]

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.y

        X_test = test_df.iloc[:, :-1]
        y_test = test_df.y


        classifier_none = LogisticRegression(solver = 'saga', penalty = 'none', max_iter = 100000, n_jobs = -1)
        search_results_none = GridSearchCV(classifier_none, param_grid = {}, scoring = metrics, refit = False, \
                                          n_jobs = -1)

        search_results_none.fit(X_train, y_train)

        no_penalty_df = pd.DataFrame().assign(C = [0], 
                                              penalty = ['none'],
                                              mean_accuracy = search_results_none.cv_results_['mean_test_accuracy'],
                                              mean_f1 = search_results_none.cv_results_['mean_test_f1'],
                                              mean_auc = search_results_none.cv_results_['mean_test_roc_auc_ovo']
                                             )
        clf = LogisticRegression(solver = 'saga', max_iter = 100000, n_jobs = -1)
        search_results = GridSearchCV(clf, params, scoring = metrics, refit = False, n_jobs = -1)
        search_results.fit(X_train, y_train)

        results = pd.DataFrame(search_results.cv_results_['params'])

        results['mean_accuracy'] = search_results.cv_results_['mean_test_accuracy']
        results['mean_f1'] = search_results.cv_results_['mean_test_f1']
        results['mean_auc'] = search_results.cv_results_['mean_test_roc_auc_ovo']

        results = results.append(no_penalty_df)
                
        opt_acc_inf = results.sort_values(by = 'mean_accuracy', ascending = False).iloc[0]
        opt_f1_inf = results.sort_values(by = 'mean_f1', ascending = False).iloc[0]
        opt_auc_inf = results.sort_values(by = 'mean_auc', ascending = False).iloc[0]
        
        if opt_acc_inf.C == 0:
            opt_acc_clf = LogisticRegression(solver = 'saga', penalty = opt_acc_inf.penalty, max_iter = 100000, \
                                             n_jobs = -1)
        else:
            opt_acc_clf = LogisticRegression(solver = 'saga', C = opt_acc_inf.C, \
                                             penalty = opt_acc_inf.penalty, max_iter = 100000, n_jobs = -1)
        
        if opt_f1_inf.C == 0:
            opt_f1_clf = LogisticRegression(solver = 'saga', penalty = opt_f1_inf.penalty, max_iter = 100000, \
                                            n_jobs = -1)
        else:
            opt_f1_clf = LogisticRegression(solver = 'saga', C = opt_f1_inf.C, \
                                            penalty = opt_f1_inf.penalty, max_iter = 100000, n_jobs = -1)
            
        if opt_auc_inf.C == 0:
            opt_auc_clf = LogisticRegression(solver = 'saga', penalty = opt_auc_inf.penalty, max_iter = 100000, \
                                             n_jobs = -1)
        else:
            opt_auc_clf = LogisticRegression(solver = 'saga', C = opt_auc_inf.C, \
                                             penalty = opt_auc_inf.penalty, max_iter = 100000, n_jobs = -1)

        
        opt_acc_clf.fit(X_train, y_train)
        opt_f1_clf.fit(X_train, y_train)
        opt_auc_clf.fit(X_train, y_train)
        
        train_score_acc = opt_acc_clf.score(X_train, y_train)
        train_score_f1 = f1_score(y_train, opt_f1_clf.predict(X_train))
        train_score_auc = roc_auc_score(y_train, opt_auc_clf.predict(X_train))
                        
        test_score_acc = opt_acc_clf.score(X_test, y_test)
        test_score_f1 = f1_score(y_test, opt_f1_clf.predict(X_test))
        test_score_auc = roc_auc_score(y_test, opt_auc_clf.predict(X_test))
        
        raw_train_arr.append([train_score_acc, train_score_f1, train_score_auc])
        raw_test_arr.append([test_score_acc, test_score_f1, test_score_auc])
        
    raw_train_arr = np.array(raw_train_arr).reshape(5, 3)
    raw_test_arr = np.array(raw_test_arr).reshape(5, 3)
    
    raw_train_df = pd.DataFrame(data = raw_train_arr, columns = ['accuracy', 'f1', 'auc'])
    raw_test_df = pd.DataFrame(data = raw_test_arr, columns = ['accuracy', 'f1', 'auc'])


    
    return raw_train_df, raw_test_df