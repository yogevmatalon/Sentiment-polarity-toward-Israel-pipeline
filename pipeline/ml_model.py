import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import (KFold, GridSearchCV)
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
import itertools
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

def create_k_fold(X_train, y_train, k=5):
    # Create a K-Fold object
    train_k_fold = pd.concat([X_train, y_train], axis=1)
    k_fold = KFold(n_splits = k, shuffle=True, random_state=1)

    # Split the train set into k sets of train and test
    k_fold.split(train_k_fold)

    # Save the indices of the train and test of for each fold
    k_fold_indices = []
    for train_index, test_index in k_fold.split(train_k_fold):
        k_fold_indices.append((train_index, test_index))
    return k_fold_indices


def manual_K_fold(X_train, y_train, model, model_name, k=5):
    accuracy = []
    auc = []
    tprs = []
    fprs = []
    predictions = []
    y_interp = []

    k_fold_indices = create_k_fold(X_train, y_train, k)

    # Set the train and test sets according to the K-Fold indices (the loop will run k times)
    for train_idx, test_idx in k_fold_indices:
        train_data = X_train.iloc[train_idx].copy()
        train_label = y_train.iloc[train_idx].copy()
        valid_data = X_train.iloc[test_idx].copy()
        valid_label = y_train.iloc[test_idx].copy()

        model = model.fit(train_data, train_label)

        # Calculte the accuracy
        accuracy.append(model.score(valid_data, valid_label))
        validation_pred = model.predict(valid_data)
        predictions.append(validation_pred)

        # Get the probabilities (confidance of the classification) - for calculating AUC
        probs = model.predict_proba(valid_data)[:, 1]

        # Calculte ROC and AUC
        fpr, tpr, thresholds = metrics.roc_curve(valid_label, probs, pos_label=1)
        auc.append(metrics.auc(fpr, tpr))
        tprs.append(tpr)
        fprs.append(fpr)

        x_range = np.arange(0.0, 1.0, 0.01)
        # calculate the interp values and append to list
        y_interp.append(np.interp(x_range, fpr, tpr))

        # Mean roc curve
        y_interp_mean = np.mean(np.transpose(np.array(y_interp)), axis=1)

    plt.figure(figsize=(10, 7))
    # plot the curves
    lw = 2
    for i in range(len(fprs)):
        plt.plot(x_range, y_interp[i], color="grey", lw=lw, alpha=0.75)
        plt.plot(x_range, y_interp_mean, color='red', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - K-fold of model {}'.format(model_name))
    plt.show()

    avg_accuracy = sum(accuracy) / float(len(accuracy))
    avg_auc = sum(auc) / float(len(auc))
    print('The average AUC over {}-folds is: {:.3f}'.format(k, avg_auc))
    print('The average accuracy over {}-folds is: {:.2f}%'.format(k, avg_accuracy * 100))
    return avg_auc, avg_accuracy


def find_number_of_components(X, pca, wanted_ratio):
    # This function find the number of components of a data that explain given variance ratio
    return np.sum(pca.explained_variance_ratio_.cumsum() < wanted_ratio) + 1

def normalize(df, type, params=None):
    '''
    :param df: data frame to normalize
    :param type: type of normalization (MinMax, scaling)
    :param params: set of parameters to use (e.g. columns means and stds) in scaling (mainly for test data that depends on train data)
    :return: normalized data frame and dictionary of data (e.g. means of train data) if required
    '''
    func = MinMax_scaling if type=='MinMax' else scaling
    if params is None:
        df, params = func(df, params if params is not None else None)
        return df, params
    else:
        df = func(df, params)
        return df

def MinMax_scaling(df, columns_data=None):
    # Function gets a data frame and apply MinMax scaling
    if columns_data:
        # Scale using pre-saved columns_data
        for col in columns_data:
            if is_numeric_dtype(df[col]):
                # Apply MinMax scaling manually
                df[col] = df[col]-columns_data[col]['min']
                df[col] = df[col]/(columns_data[col]['max']-columns_data[col]['min'])
        return df
    else:
        # Scale the data independently
        columns_data = {}
        scaler = preprocessing.MinMaxScaler()
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                columns_data[col] = {}
                columns_data[col]['max'] = df[col].max()
                columns_data[col]['min'] = df[col].min()
        # Transform data
        cols = df.columns
        relevant_cols = [col for col in columns_data]
        index_col = df.index
        df[relevant_cols] = scaler.fit_transform(df[relevant_cols])
        df = pd.DataFrame(df, columns=cols)
        df.index = index_col
        return df, columns_data

def scaling(df, columns_data = False, center = True, std = True):
    # Gets a data frame and apply standard scaling
    # Columns_data is an indicator if this df should be scaled by pre-saved data (e.g. for test data, using train data)
        # If true, the function also return a dictionary with key = col name, value = {'avg':mean, 'std':std}
    if not columns_data:
        columns_data = {}
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                columns_data[col] = {}
                columns_data[col]['avg'] = df[col].mean()
                columns_data[col]['std'] = df[col].std()
        cols = df.columns
        index_col = df.index
        df = preprocessing.scale(df, with_mean = center, with_std = std)  # Z scaling (minus average, and divide by the std)
        df = pd.DataFrame(df, columns=cols)
        df.index = index_col
        return df, columns_data
    else:
        # Use existing scaling data
        for col in columns_data:
            # Apply scaling
            df[col] = df[col]-columns_data[col]['avg'] if center else df[col]
            df[col] = df[col]/columns_data[col]['std'] if std else df[col]
        return df

def remove_categorical_features(df):
    # The function gets a df and return a new df with only numerical columns
    print('Removing categorical features')
    all_cols = df.columns
    df = df._get_numeric_data()
    return df

def evaluate_multi_class_algo(x_train, y_train, x_test, y_test, models, config):
    # Use models to predict the test
    prob_train = {}
    prob_test = {}
    prob_columns = []  # Will save the names of the prediction columns

    for class_i in models:
        # Predict proba
        prob_train['prob_{}'.format(class_i)] = models[class_i].predict_proba(x_train)[:, 1]
        prob_test['prob_{}'.format(class_i)] = models[class_i].predict_proba(x_test)[:, 1]

        # Save columns names
        prob_columns.append('prob_{}'.format(class_i))

    # Build data frames for all test predictions, from all 3 models
    predictions_train = pd.DataFrame.from_dict(prob_train, orient='columns')
    predictions_test = pd.DataFrame.from_dict(prob_test, orient='columns')

    ########## Handle the TRAIN SET ##########
    # Extract the max probability and therefore the winner classification
    predictions_train['sum_probs'] = predictions_train[prob_columns].sum(axis=1)

    # Normalize probabilities
    for i in prob_columns:
        predictions_train[i] = predictions_train[i] / predictions_train['sum_probs']

    # Drop sum_probs column
    predictions_train = predictions_train.drop(['sum_probs'], axis=1)
    predictions_train['max_prob'] = predictions_train[prob_columns].max(axis=1)

    print('predictions_train:')
    print(predictions_train.head())
    
    predictions_train['prediction'] = predictions_train[prob_columns].idxmax(axis=1) # We get prob_i answer

    print('prediction columns :')
    print(predictions_train[['prob_-1','prob_0','prob_1','prediction']].head())

    # Convert to number
    predictions_train['prediction'] = predictions_train['prediction'].apply(lambda x: float(x.split('_')[1]))
    predictions_train['prediction'] = predictions_train['prediction'].astype(float)

    ########## On the TEST SET ##########
    predictions_test['sum_probs'] = predictions_test[prob_columns].sum(axis=1)

    # Normalize probabilities
    for i in prob_columns:
        predictions_test[i] = predictions_test[i] / predictions_test['sum_probs']

    # Drop sum_probs column
    predictions_test = predictions_test.drop(['sum_probs'], axis=1)
    predictions_test['max_prob'] = predictions_test[prob_columns].max(axis=1)
    predictions_test['prediction'] = predictions_test[prob_columns].idxmax(axis=1) # We get prob_i answer
    # Convert to number
    predictions_test['prediction'] = predictions_test['prediction'].apply(lambda x: float(x.split('_')[1]))
    predictions_test['prediction'] = predictions_test['prediction'].astype(float)

    predictions_train['true_value'] = y_train.reset_index(drop=True)
    predictions_test['true_value'] = y_test.reset_index(drop=True)

    # # Change predictions according to thresholds
    # predictions_train['prediction'] = predictions_train[['max_prob', 'prediction']].apply(
    #     lambda x: x.prediction if x.max_prob > config['class_threshold'] else 'Unknown', axis=1)
    # predictions_test['prediction'] = predictions_test[['max_prob', 'prediction']].apply(
    #     lambda x: x.prediction if x.max_prob > config['class_threshold'] else 'Unknown', axis=1)

    # Calculate weighted prediction - weighted average over support probabilities
    multipliers = [-1,0,1]
    predictions_train['prediction_weighted'] = predictions_train[prob_columns[0]] * multipliers[0] + predictions_train[prob_columns[1]] * multipliers[1] + predictions_train[prob_columns[2]] * multipliers[2]
    predictions_test['prediction_weighted'] = predictions_test[prob_columns[0]] * multipliers[0] + predictions_test[prob_columns[1]] * multipliers[1] + predictions_test[prob_columns[2]] * multipliers[2]

    # Save the test and train sets sizes
    n_train, n_test = predictions_train.shape[0], predictions_test.shape[0]

    # Filter possible 'Unknown' classifications
    predictions_train = predictions_train[predictions_train.prediction.isin([-1,0,1])]
    predictions_test = predictions_test[predictions_test.prediction.isin([-1,0,1])]

    # Save number of unclassified records - obsolete
    uncl_train, uncl_test = n_train-predictions_train.shape[0], n_test-predictions_test.shape[0]
    # print(f'Unclassified tweets: train-{uncl_train}, test-{uncl_test}')

    # Evaluate
    labels = {'support': {'names': ['Anti-Israel', 'Neutral', 'Pro-Israel'], 'values': [-1, 0, 1]},
              'user_support': {'names': ['Anti-Israel', 'Neutral', 'Pro-Israel'], 'values': [-1, 0, 1]},
              'user_support_network': {'names': ['Anti-Israel', 'Neutral', 'Pro-Israel'], 'values': [-1, 0, 1]},
              'relevance': {'names': ['Not relevant', 'Relevant', 'Ignore'], 'values': [0, 1, 2]}}
    plot_confusion_matrix(list(predictions_train.prediction), list(predictions_train.true_value), '{} SUMMARY'.format(config['model']),
                                   'train', labels[config['target']])
    plot_confusion_matrix(list(predictions_test.prediction), list(predictions_test.true_value), '{} SUMMARY'.format(config['model']), 'test',
                                   labels[config['target']])

    # Calculate accuracy
    accuracy_train = metrics.accuracy_score(list(predictions_train.true_value), list(predictions_train.prediction),
                                            normalize=True)
    accuracy = metrics.accuracy_score(list(predictions_test.true_value), list(predictions_test.prediction),
                                      normalize=True)

    # Calculate adjusted accuracy
    adj_accuracy_train = calc_adj_accuracy(list(predictions_train.true_value), list(predictions_train.prediction))
    adj_accuracy_test = calc_adj_accuracy(list(predictions_test.true_value), list(predictions_test.prediction))

    print('############################################################################')
    return {'accuracy_train': accuracy_train, 'accuracy_test': accuracy, 'unclassified_train': np.round(uncl_train/n_train,4),
            'unclassified_test': np.round(uncl_test/n_test,4),'adj_accuracy_test': adj_accuracy_test, 'adj_accuracy_train': adj_accuracy_train}

def apply_PCA(df, label_name, features, variance_thresh):
    '''
    :param df: data frame to apply pca to
    :param variance_thresh: a threshold for %of variance to cover (will affect the number of components retrieved)
    :return: new data frame with principal components, according to the variance_thresh parameter
    '''
    # Z-normalization on the data
    df_PCA = df[features].copy()
    df_PCA = preprocessing.scale(df_PCA)  # scaling before PCA is mandatory. We used 'z' scaling (minus average, and divide by the std)
    # first, we will choose all the features as the number of the component
    pca = PCA(n_components=df_PCA.shape[1])
    # We will "train" our PCA on the given train data
    pca.fit(df[features])
    # we will print the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    plt.title('PCA plot')
    plt.ylabel('Part of explained variance')
    plt.xlabel('Number of components')
    plt.plot(explained_variance_ratio.cumsum())
    plt.show()

    # Variance explained threshold = variance_thresh
    comp = find_number_of_components(df_PCA, pca, variance_thresh)
    # we will apply it on the same train data
    reduced_data = pca.transform(df_PCA)[:, :comp]
    # Create a new train set with the principal components
    df_PCA = pd.DataFrame(reduced_data, columns=['PC{}'.format(i + 1) for i in range(comp)], index=df.index)
    df_PCA = pd.concat([df_PCA, df[label_name], df.text, df.tokenized_text], axis=1)
    return df_PCA, pca

def run_ML_model(model_object, model_name, relevant_class, X_train, X_test, y_train, y_test, tuned_parameters, cv = 5, scoring = 'roc_auc', check_on_test = False, return_stats = False, run_manual_kfold=True):
    # Convert y_train and y_test to 1 vs All classification problem
    y_train = y_train.copy().apply(lambda x: 1 if x == relevant_class else 0)
    y_test = y_test.copy().apply(lambda x: 1 if x == relevant_class else 0)

    if model_object is GaussianNB:
        print('Classes priors:', [np.round(x,4) for x in tuned_parameters])
        clf_model = model_object(priors = tuned_parameters)
    else:
        if model_object is RandomForestClassifier:
            tuned_parameters['class_weight'] = ['balanced']
            clf_model = GridSearchCV(model_object(), tuned_parameters, cv=cv, scoring=scoring) if 'kernel' not in tuned_parameters else GridSearchCV(model_object(), tuned_parameters, cv=cv, scoring=scoring)

        else:
            clf_model = GridSearchCV(model_object(), tuned_parameters, cv=cv, scoring=scoring) if 'kernel' not in tuned_parameters else GridSearchCV(model_object(), tuned_parameters, cv=cv, scoring=scoring)

    # Train model on the entire train set
    clf_model.fit(X_train, y_train)

    # Run the model with the best parameters
    if model_object is not GaussianNB:
        params = clf_model.best_params_
        print('Best parameters are: {}'.format(params))
        print('Model AUC: {:.3f}'.format(clf_model.best_score_))
        man_model = model_object(**params if model_object is not GaussianNB else tuned_parameters)
        auc_k_fold, avg_accuracy = manual_K_fold(X_train, y_train, man_model, model_name) if run_manual_kfold else clf_model.best_score_, None
    else:
        auc_k_fold, avg_accuracy = manual_K_fold(X_train, y_train, clf_model, model_name) if run_manual_kfold else clf_model.best_score_, None

    # Check performance on the test set
    # train the classifier over all train data
    clf_model = model_object(**params) if model_object is not GaussianNB else model_object(priors = tuned_parameters)
    # Fit the model on the entire train set
    clf_model.fit(X_train, y_train)

    if check_on_test:
        model_stats = generate_model_stats(clf_model, model_name, X_train, y_train, X_test, y_test, auc_k_fold, avg_accuracy, True)
    if not return_stats:
        return clf_model
    else:
        return clf_model, model_stats


def generate_model_stats(model, name, X_train, y_train, X_test, y_test, auc_k_fold, avg_accuracy_k_folds, MSE=False, plots=True):
    '''
    :param model: ML model to compute stats for
    :param name: ML model name
    :param X_train: training features
    :param y_train: training label
    :param X_test: test features
    :param y_test: test label
    :param auc_k_fold: the auc of the k-fold
    :param avg_accuracy_k_folds: optional, the avg_accuracy of the manual k-fold
    :param MSE: bool parameter - if to calculate MSE (both for train and test)
    :param plots: bool, if to plot confusion matrix and ROC curve
    :return: stats dictionary
    '''
    # prediction of train set
    probs_train = model.predict_proba(X_train)[:,1]
    pred_train = model.predict(X_train)

    # print MSE on the train set if required
    if MSE:
        mse_train = metrics.mean_squared_error(probs_train, y_train)
        print("The MSE on the train set: {:.4f}".format(mse_train))

    # Predict values + probs on test set
    pred_test = model.predict(X_test)
    probs_test = model.predict_proba(X_test)[:,1]

    # print MSE on the test set if required
    if MSE:
        mse_test = metrics.mean_squared_error(probs_test, y_test)
        print("The MSE on the test set: {:.4f}".format(mse_test))

    # Calculate tpr, fpr and auc
    from scipy.interpolate import spline
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs_test, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    if plots:
        # Plot confusion matrix
        plot_confusion_matrix(pred_test, y_test, name, 'test')
        # Plot ROC curve
        plot_ROC(fpr, tpr, auc, name)

    # Calculate accuracy
    accuracy_train = metrics.accuracy_score(y_train, pred_train, normalize=True)
    accuracy_test = metrics.accuracy_score(y_test, pred_test, normalize=True)

    # Save stats and return it
    stats = {'name': name, 'fpr': fpr, 'tpr': tpr, 'auc_test': auc,
             'accuracy_test': accuracy_test, 'avg_accuracy_k_folds': avg_accuracy_k_folds, 'accuracy_train':accuracy_train,
             'auc_k_fold': auc_k_fold, 'mse_train': mse_train, 'mse_test': mse_test}
    return stats


def regression_rf(model_object, model_name, X_train, X_test, y_train, y_test, tuned_parameters, cv = 5, scoring = 'neg_mean_squared_error', check_on_test = False):
    clf_model = GridSearchCV(model_object(), tuned_parameters, cv=cv, scoring=scoring)
    clf_model.fit(X_train, y_train)

    # Run the model with the best parameters
    params = clf_model.best_params_
    print('Best parameters are: {}'.format(params))
    print('Model MSE: {:.3f}'.format(clf_model.best_score_))

    # man_forest = model_object(**params)

    # Check performance on the test set
    # train the classifier over all train data
    clf_model = model_object(**params)
    # clf_model = model_object(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], n_estimators=params['n_estimators'], criterion=params['criterion'],random_state=1)
    # fit the model on the entire train set
    clf_model.fit(X_train, y_train)
    pred_train = clf_model.predict(X_train)
    mse_train = metrics.mean_squared_error(pred_train, y_train)
    print("The MSE on the train set: {:.4f}".format(mse_train))
    if check_on_test:
        # predict values + probs on test set
        pred_test = clf_model.predict(X_test)
        # print MSE on the test set if required
        mse_test = metrics.mean_squared_error(pred_test, y_test)
        print("The MSE on the test set: {:.4f}".format(mse_test))

    return clf_model

def show_reg_result(rf_regression, x, y, evaluation,config):
    regressor_test = pd.DataFrame(rf_regression.predict(x), y)
    regressor_test = regressor_test.reset_index()
    regressor_test.columns = ['label', 'prediction']
    regressor_test = regressor_test.sort_values(by='prediction')
    colors = ['red', 'blue', 'green']
    fig = plt.figure()
    if config['target']=='virality':
        plt.plot(x, x, '-', color='black')
        plt.scatter(regressor_test.prediction, regressor_test.label)
    else:
        plt.scatter(regressor_test.prediction, regressor_test.label, c=regressor_test.label,
                cmap=matplotlib.colors.ListedColormap(colors))
    fig.suptitle('{} evaluation'.format(evaluation), fontsize=20)
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Label', fontsize=15)
    plt.show()

def run_XGBoost(X_train, X_test, y_train, y_test, params):
    # Convert classes to be 0-2
    label_converter = {-1: 0, 0: 1, 1: 2}
    Y_train = y_train.apply(lambda x: label_converter[x])
    Y_test = y_test.apply(lambda x: label_converter[x])

    grid_search = GridSearchCV(xgb.XGBClassifier(params['model_params']), param_grid=params['grid_params'], cv=5,
                               verbose=0, n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    print('Best estimator', grid_search.best_estimator_)

    best_params = grid_search.best_params_
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, Y_train)

    # Predict
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    targets_labels = ['Negative (-1)', 'Neutral(0)', 'Positive (1)']
    print(classification_report(Y_test, pred_test, target_names=targets_labels))

    # Evaluation
    accuracy_train = metrics.accuracy_score(Y_train, pred_train, normalize=True)
    accuracy_test = metrics.accuracy_score(Y_test, pred_test, normalize=True)
    adj_accuracy_train = calc_adj_accuracy(Y_train, pred_train, 1)  # neutral class is 1 in xgboost
    adj_accuracy_test = calc_adj_accuracy(Y_test, pred_test, 1) # neutral class is 1 in xgboost

    # Print results
    print('XGBoost accuracy on the test set:', accuracy_test)
    print('XGBoost adjusted accuracy on the test set:',adj_accuracy_test)

    # Plot confusion matrix
    labels = {'support': {'names': ['Anti-Israel', 'Neutral', 'Pro-Israel'], 'values': [0, 1, 2]},
              'user_support': {'names': ['Anti-Israel', 'Neutral', 'Pro-Israel'], 'values': [0, 1, 2]},
              'user_support_network': {'names': ['Anti-Israel', 'Neutral', 'Pro-Israel'], 'values': [0, 1, 2]}}

    plot_confusion_matrix(pred_train, Y_train,'Support SUMMARY', 'train', labels['support'])
    plot_confusion_matrix(pred_test, Y_test,'Support SUMMARY', 'test', labels['support'])

    model_stats = {'name': 'XGBoost', 'summary': { 'accuracy_test': accuracy_test, 'accuracy_train': accuracy_train,
                   'adj_accuracy_train':adj_accuracy_train, 'adj_accuracy_test': adj_accuracy_test }}
    return model, model_stats

def run_LightGBM(X_train, X_test, y_train, y_test, params):
    mdl = lgb.LGBMClassifier()

    # Create the grid
    grid = GridSearchCV(mdl, param_grid=params, cv=5, n_jobs=-1)
    # Run the grid
    grid.fit(X_train, y_train)

    # Print the best parameters found
    best_params = grid.best_params_
    print('Best params:', best_params)
    print('Accuracy on test:', grid.best_score_)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    # Predict
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    targets_labels = ['Anti (-1)', 'Neutral(0)', 'Pro (1)']
    print(classification_report(y_test, pred_test, target_names=targets_labels))

    # Evaluation
    accuracy_train = metrics.accuracy_score(y_train, pred_train, normalize=True)
    accuracy_test = metrics.accuracy_score(y_test, pred_test, normalize=True)
    adj_accuracy_train = calc_adj_accuracy(y_train, pred_train)
    adj_accuracy_test = calc_adj_accuracy(y_test, pred_test)

    # Print results
    print('LightGBM accuracy on the test set:', accuracy_test)
    print('LightGBM adjusted accuracy on the test set:',adj_accuracy_test)

    # Plot confusion matrix
    labels = {'support': {'names': ['Anti-Israel', 'Neutral', 'Pro-Israel'], 'values': [-1, 0, 1]},
              'user_support': {'names': ['Anti-Israel', 'Neutral', 'Pro-Israel'], 'values': [-1, 0, 1]},
              'user_support_network': {'names': ['Anti-Israel', 'Neutral', 'Pro-Israel'], 'values': [-1, 0, 1]}}

    plot_confusion_matrix(pred_train, y_train,'Support SUMMARY', 'train', labels['support'])
    plot_confusion_matrix(pred_test, y_test,'Support SUMMARY', 'test', labels['support'])

    model_stats = {'name': 'LightGBM', 'summary': { 'accuracy_test': accuracy_test, 'accuracy_train': accuracy_train,
                   'adj_accuracy_train':adj_accuracy_train, 'adj_accuracy_test': adj_accuracy_test }}
    return model, model_stats


def plot_ROC(fpr, tpr, auc, model_name=''):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=80)
    plt.plot(fpr, tpr, markevery= 100,lw=2, label='ROC Curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_facecolor('w')
    fig.savefig('ROC Curve - {}.png'.format(model_name), bbox_inches='tight')
    #plt.title('ROC Curve - {}'.format(model_name))
    plt.show()
    print('AUC achieved: {:.2f}'.format(auc))


def plot_multiple_ROC(models_res):
    '''
    :param models_res:  dictionary, with keys (at least) - 'fpr', 'tpr', 'name', 'auc'
    :return: Function plot several roc curves, mainly for models comparison
    '''
    fig,ax = plt.subplots(1, 1, figsize=(10, 7), dpi=80)
    rocs = []
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for model in models_res:
        plt.plot(model['fpr'], model['tpr'], lw=2, label='{} (AUC = {:.2f})'.format(model['name'], model['auc_test']),
                 figure=fig)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_facecolor('w')
    fig.savefig('Multiple ROC Curve - Relevance.png', bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(predict, y, model_name, validation_type, labels_dict = {'names': ['0','1'], 'values': [0,1]}):
    plt.figure(figsize=(8,8))
    labels = labels_dict['names']
    cm = metrics.confusion_matrix(y, predict, labels=labels_dict['values'])  # display absolute confusion matrix
    title = 'Confusion matrix - {}'.format(model_name)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print("Accuracy on {}: {:.2f}".format(validation_type, metrics.accuracy_score(y, predict, normalize=True)))

def calc_adj_accuracy(y_true, y_pred, neutral_class=0):
    '''
        Function gets 2 pandas Series and calculate smart accuracy of them.
        neutral_class is the index of the neutral class. It equals zero, except for XGBoost (1)
    '''
    # Smart accuracy:
    # In case that the max probability is neutral, each possible error is evenly important (neutral-positive/ neutral-negative)
    # In case that the max probability is positive/negative, neutral error is 0.5 error, while the contrast is 1 error.
    n_total = len(y_true)
    agg_erros = 0
    for i in range(n_total):
        if y_true[i] == neutral_class:
            # True label is neutral
            agg_erros+= 1 if y_pred[i] != neutral_class else 0

        elif y_pred[i]!= y_true[i]:
            # True label not is neutral, and the algo predicted other class
            agg_erros += 1.5 if y_pred[i] != neutral_class else 0.5

        # Else - no need to add error
    # Return accuracy
    return agg_erros/n_total


