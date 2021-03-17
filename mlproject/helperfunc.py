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
    # Initiate classifier, C values, and gamma values
    clf = SVC()
    C_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    
    # 'auto' means 1/(n_features)
    gamma_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 'auto']

    search_params = {
        'C': C_list,
        'gamma': gamma_list
    }
    
    # Set metrics
    metrics = ['accuracy', 'f1', 'roc_auc']

    raw_train_arr = []
    raw_test_arr = []
    
    # Over five trials
    for i in range(5):
        
        # Train test split data
        X_train, X_test, y_train, y_test = train_test_split(scaled_df.iloc[:, :-1], scaled_df.y, train_size = 5000)
        
        # Init GridSearch
        search_results = GridSearchCV(clf, search_params, scoring = metrics, refit = False)
        
        # Run GridSearch
        search_results.fit(X_train, y_train)
        
        # Get results
        results = pd.DataFrame(search_results.cv_results_['params'])

        results['mean_accuracy'] = search_results.cv_results_['mean_test_accuracy']
        results['mean_f1'] = search_results.cv_results_['mean_test_f1']
        results['mean_auc'] = search_results.cv_results_['mean_test_roc_auc']
        
        # Get optimal classifier using results dataframe
        opt_acc_inf = results.sort_values(by = 'mean_accuracy', ascending = False).iloc[0]
        opt_f1_inf = results.sort_values(by = 'mean_f1', ascending = False).iloc[0]
        opt_auc_inf = results.sort_values(by = 'mean_auc', ascending = False).iloc[0]
        
        # Init optimal classifiers
        opt_acc_clf = SVC(C = opt_acc_inf.C, gamma = opt_acc_inf.gamma)
        opt_f1_clf = SVC(C = opt_f1_inf.C, gamma = opt_f1_inf.gamma)
        opt_auc_clf = SVC(C = opt_auc_inf.C, gamma = opt_auc_inf.gamma)
        
        # Fit to train
        opt_acc_clf.fit(X_train, y_train)
        opt_f1_clf.fit(X_train, y_train)
        opt_auc_clf.fit(X_train, y_train)
        
        # Get train and test metrics
        train_score_acc = opt_acc_clf.score(X_train, y_train)
        train_score_f1 = f1_score(y_train, opt_f1_clf.predict(X_train))
        train_score_auc = roc_auc_score(y_train, opt_auc_clf.predict(X_train))
                        
        test_score_acc = opt_acc_clf.score(X_test, y_test)
        test_score_f1 = f1_score(y_test, opt_f1_clf.predict(X_test))
        test_score_auc = roc_auc_score(y_test, opt_auc_clf.predict(X_test))
        
        # Append to results
        raw_train_arr.append([train_score_acc, train_score_f1, train_score_auc])
        raw_test_arr.append([test_score_acc, test_score_f1, test_score_auc])
        
                
    
    raw_train_arr = np.array(raw_train_arr).reshape(5, 3)
    raw_test_arr = np.array(raw_test_arr).reshape(5, 3)
    
    raw_train_df = pd.DataFrame(data = raw_train_arr, columns = ['accuracy', 'f1', 'auc'])
    raw_test_df = pd.DataFrame(data = raw_test_arr, columns = ['accuracy', 'f1', 'auc'])
    
    # Return results
    return raw_train_df, raw_test_df

def GridSearch_random_forest(X_train, y_train):
    # Encode as float32
    X_train = X_train.to_numpy().astype('float32')
    y_train = y_train.to_numpy().astype('float32')
    
    
    # Init Kfolds
    folds = KFold(n_splits = 5)
    
    # Init hyperparam vals
    n_estimators_lst = [128, 256, 512, 1024]
    max_features_lst = ['sqrt', 'log2']
        
    fin_arr = []    
    
    # Run GridSearch for all hyperparam combos
    for n_estimators in n_estimators_lst:
        
        for max_features in max_features_lst:
            
            # Init clf
            clf = RandomForestClassifier(n_estimators = n_estimators, max_features = max_features)
            
            predicted_y = []
            true_y = []
            # Run CV and calc metrics
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
    # Create final dataframe from GridSearch results    
    fin_arr = np.array(fin_arr).reshape((len(n_estimators_lst) * len(max_features_lst)), 5)
    
    columns = ['n_estimators', 'max_features', 'mean_accuracy', 'mean_f1', 'mean_auc']
    
    results = pd.DataFrame(data = fin_arr, columns = columns)
    results.n_estimators = results.n_estimators.astype(int)
    
    

    return results

def run_random_forest(scaled_df):
    raw_train_arr = []
    raw_test_arr = []
    # Over five trials
    for i in range(5):
        
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(scaled_df.iloc[:, :-1], scaled_df.y, train_size = 5000)
        
        # Run GridSearch
        search_results = GridSearch_random_forest(X_train, y_train)
                
        results = search_results
        # Get optimal clfs using gridsearch results
        opt_acc_inf = results.sort_values(by = 'mean_accuracy', ascending = False).iloc[0]
        opt_f1_inf = results.sort_values(by = 'mean_f1', ascending = False).iloc[0]
        opt_auc_inf = results.sort_values(by = 'mean_auc', ascending = False).iloc[0]
        
        # Init optimal clfs
        opt_acc_clf = RandomForestClassifier(n_estimators = opt_acc_inf.n_estimators,
                                             max_features = opt_acc_inf.max_features)

        opt_f1_clf = RandomForestClassifier(n_estimators = opt_f1_inf.n_estimators,
                                             max_features = opt_f1_inf.max_features)

        opt_auc_clf = RandomForestClassifier(n_estimators = opt_auc_inf.n_estimators,
                                             max_features = opt_auc_inf.max_features)
        
        # Encode as float32 for cuML
        X_train_np = X_train.to_numpy().astype('float32')
        y_train_np = y_train.to_numpy().astype('float32')
        
        X_test_np = X_test.to_numpy().astype('float32')
        y_test_np = y_test.to_numpy().astype('float32')
        
        # Fit clfs
        opt_acc_clf.fit(X_train_np, y_train_np)
        opt_f1_clf.fit(X_train_np, y_train_np)
        opt_auc_clf.fit(X_train_np, y_train_np)
        
        # Get train and test metrics
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
    
    # Init metrics
    metrics = ['accuracy', 'f1', 'roc_auc_ovr']
    
    # Set c vals and penalty
    C_vals = range(-8, 5)
    C_vals = [10 ** val for val in C_vals]
    
    penalty = ['none', 'l1', 'l2']
    
    # Init params
    params = {'penalty' : penalty, 'C' : C_vals}


    # Over five trials
    for i in range(5):
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(scaled_df.iloc[:, :-1], scaled_df.y, train_size = 5000)
        
        # Init clf
        clf = LogisticRegression()
        
        # Init gridsearch and run
        search_results = GridSearchCV(clf, params, scoring = metrics, refit = False)
        search_results.fit(X_train, y_train)
        
        # Get results and organize
        results = pd.DataFrame(search_results.cv_results_['params'])

        results['mean_accuracy'] = search_results.cv_results_['mean_test_accuracy']
        results['mean_f1'] = search_results.cv_results_['mean_test_f1']
        results['mean_auc'] = search_results.cv_results_['mean_test_roc_auc_ovr']

        # Get optimal clfs
        opt_acc_inf = results.sort_values(by = 'mean_accuracy', ascending = False).iloc[0]
        opt_f1_inf = results.sort_values(by = 'mean_f1', ascending = False).iloc[0]
        opt_auc_inf = results.sort_values(by = 'mean_auc', ascending = False).iloc[0]
        
        # Init optimal clfs
        opt_acc_clf = LogisticRegression(C = opt_acc_inf.C, penalty = opt_acc_inf.penalty, max_iter = 100000)
        opt_f1_clf = LogisticRegression(C = opt_f1_inf.C, penalty = opt_f1_inf.penalty, max_iter = 100000)
        opt_auc_clf = LogisticRegression(C = opt_auc_inf.C, penalty = opt_auc_inf.penalty, max_iter = 100000)

        # Fit clfs
        opt_acc_clf.fit(X_train, y_train)
        opt_f1_clf.fit(X_train, y_train)
        opt_auc_clf.fit(X_train, y_train)
        
        # Get train and test metrics
        train_score_acc = opt_acc_clf.score(X_train, y_train)
        train_score_f1 = f1_score(y_train, opt_f1_clf.predict(X_train))
        train_score_auc = roc_auc_score(y_train, opt_auc_clf.predict(X_train))
                        
        test_score_acc = opt_acc_clf.score(X_test, y_test)
        test_score_f1 = f1_score(y_test, opt_f1_clf.predict(X_test))
        test_score_auc = roc_auc_score(y_test, opt_auc_clf.predict(X_test))
        
        raw_train_arr.append([train_score_acc, train_score_f1, train_score_auc])
        raw_test_arr.append([test_score_acc, test_score_f1, test_score_auc])
    
    # Create dataframe from results
    raw_train_arr = np.array(raw_train_arr).reshape(5, 3)
    raw_test_arr = np.array(raw_test_arr).reshape(5, 3)
    
    raw_train_df = pd.DataFrame(data = raw_train_arr, columns = ['accuracy', 'f1', 'auc'])
    raw_test_df = pd.DataFrame(data = raw_test_arr, columns = ['accuracy', 'f1', 'auc'])


    # Return results
    return raw_train_df, raw_test_df