'''
This script apply the support for Israel classification algorithm to new unseen and unlabeled data.
Input: a relevant tweet or batch of tweets
Process: apply pre-process, feature engineering, and classification (-1,0,1,'unknown')
Output: a classification per input: -1 is anti Israel, 0 neutral, 1 pro Israel or 'unknown'.
1. Using the Pipeline.py for pre-process, feature engineering (NLP, entities, user features) and feature selection
    Dominant words list is taken from learning
    Features are eventually filtered according to the final features in learning
2. Output support label data
'''

#Imports packages
import pandas as pd
import numpy as np
import os
import pickle
import multiprocessing as mp
import xgboost as xgb

#Imports modules
os.chdir('../pipeline')
import nlp_features
import pipeline
import network_features
import user_bio

def predict(df, config):
    os.chdir('../support algo')
    # Taking only needed features
    df_full_features = df.copy()
    with open('support_features_list.pickle', 'rb') as handle:
        train_features = pickle.load(handle)

    # Validate no labels in the train_features list
    for col in ['Support','support','Relevance(0,1,2)']:
        if col in train_features:
            train_features.remove(col)

    # Filter df features
    df = df[train_features]
    print('Support train features:')
    print(pd.DataFrame(train_features, columns=['features']))

    # # Saving df
    # if not config['load_df_from_pickle']:
    #     # df_full_features is a df with all features created for support algorithm, without feature selection
    #     print('Export the data with all features required for the support algorithm, df_pipeline_support_all_features.tsv')
    #     df_full_features.to_csv('../pipeline/predictions/df_pipeline_support_all_features.tsv', header=True, sep='\t')

    # Search for np.nan
    # for col in df.columns:
    #     temp = df[col].apply(lambda x: 1 if x == np.nan else 0)
    #     print('{}: {}'.format(col, temp.sum())) if temp.sum() > 0 else None
    #     print('\nCheck for NaNs: ' , df.isnull().values.any())

    # Predict
    if config['support_model'] == 'RF':
        # Loading model
        with open('../../data/classifiers/support_models_Random Forest.pickle', 'rb') as handle:
            model = pickle.load(handle)

    elif config['support_model'] == 'XGBoost':
        # Loading model
        with open('../../data/classifiers/support_XGBoost_model.pickle', 'rb') as handle:
            model = pickle.load(handle)

    # Predict label
    df = pred_label(df, config, model)
    return df, df_full_features

def pred_label(df,config,model):
    if config['support_model'] == 'RF':
        # Use models to predict the support
        probs = {}
        prob_columns = []  # Will save the names of the prediction columns

        for class_i in model:
            probs['prob_{}'.format(class_i)] = model[class_i].predict_proba(df)[:, 1]

            # Save columns names
            prob_columns.append('prob_{}'.format(class_i))

        # Build data-frames for all predictions, from all 3 models
        predictions = pd.DataFrame.from_dict(probs, orient='columns')

        # Sum of "probs"
        predictions['sum_probs'] = predictions[prob_columns].sum(axis=1)

        # Normalize probabilities
        for i in prob_columns:
            predictions[i] = predictions[i] / predictions['sum_probs']

        # Extract the max probability and therefore the winner classification
        predictions = predictions.drop(['sum_probs'], axis=1)
        predictions['max_prob'] = predictions.max(axis=1)

        predictions['prediction'] = predictions[prob_columns].idxmax(axis=1)  # We get prob_i answer
        # Convert to number
        predictions['prediction'] = predictions['prediction'].apply(lambda x: float(x.split('_')[1]))
        predictions['prediction'] = predictions['prediction'].astype(float)

        # Change predictions according to thresholds
        predictions['prediction'] = predictions[['max_prob', 'prediction']].apply(
            lambda x: x.prediction if x.max_prob > config['class_threshold'] else 'Unknown', axis=1)

        # Calculate weighted prediction - weighted average over support probabilities
        multipliers = [-1, 0, 1]
        predictions['prediction_weighted'] = predictions[prob_columns[0]] * multipliers[0] + predictions[prob_columns[1]] * multipliers[1] + predictions[prob_columns[2]] * multipliers[2]

        n = predictions.shape[0]
        k = predictions[predictions.prediction != 'Unknown'].shape[0]
        print(f'Unclassified tweets: {n-k}')

        df['support'] = predictions['prediction']
        if config.get('predictions_weighted', True):
            df['support_weighted'] = predictions['prediction_weighted']

    elif config['support_model'] == 'XGBoost':
        # Predict
        df['support'] = model.predict(df)
        print('support')
        print(df.support.value_counts())
        print(df.support.head(n=5))
        # Convert support values from 0,1,2 to -1,0,1
        label_converter = {-1: 0, 0: 1, 1: 2}
        df['support'] = df['support'].apply(lambda x: label_converter[x])
        print(df.support.value_counts())

        print('support_weighted')
        df['support_weighted'] = model.predict_proba(df)
        print(df.support_weighted.value_counts())
        print(df.support_weighted.head(n=5))
        df['support_weighted'] = df['support_weighted'].apply(lambda probs: -1*probs[0]+1*probs[2])
        print(df.support_weighted.head(n=5))
        print(df.support_weighted.value_counts())

    return df

def add_support(df,data):
    # Reset index in order to sync between rows
    df = df.reset_index(drop=True)
    # List of features to save in the 'data' object
    irrelevant_data = data[data['relevance'] == 2].reset_index(drop=True)
    relevant_data = data[data['relevance'] != 2].reset_index(drop=True)
    # Add support prediction to the relevant features only
    relevant_data = pd.concat([relevant_data, df[['support','support_weighted']]], axis=1)
    # Return concat of relevant and irrelevant tweets. Support for irrelevant tweets will np.nan
    return pd.concat([relevant_data, irrelevant_data], axis=0).reset_index(drop=True)


def adding_features(df, config):
    # Remove dominant words features
    print('Removing dominant words features of previous algorithm')
    df = nlp_features.remove_dominant_words(df,config)
    print(' - COMPLETED')

    # Adding features which are required for the support algorithm
    print('Building features for support algo')

    # Adding country support feature
    df = nlp_features.get_country_support(df) if config['country_support'] else df

    # Adding word type feature
    df = nlp_features.creating_word_type(df) if config['word_type'] else df

    # Add user features and user bio features
    if config.get('user_features', False) or config.get('user_bio', False):
        # Load users db in order to import user metadata and features
        users_db = pd.read_csv('../../data/users_DB_no_location.csv')
        # Remove unwanted features
        users_db = users_db.iloc[:, 2:]
        # Remove duplicates
        users_db = users_db.drop_duplicates(subset=['screen_name'], keep='first')

    # Add user metadata (users_db) features
    df = pipeline.add_user_features(df, config, users_db) if config.get('user_features',False) else df

    # Add network features
    df = network_features.add_network_features(df, config.get('load_network_data', True)) if config.get('network_features', False) else df

    # Add user bio features
    df = pipeline.add_user_bio_features(df, config, users_db) if config.get('user_bio',False) else df

    # Add time and event features
    df = pipeline.add_time_and_event(df, config) if config.get('time_and_event', False) else df

    # Adding sentiment feature
    print('- Tweet sentiment')
    df['tweet_sentiment'] = df.text.apply(lambda x: nlp_features.analyze_sentiment(x))

    # Add dominant words related to the support algorithm
    df, dominant_keywords = nlp_features.dominant_keywords(df, config) if config['dominant_keywords'] else df

    # Adding emotion feature
    df = nlp_features.emotion_extraction(df) if config['emotion'] else df
    print(' - COMPLETED')
    return df

def adding_features_mp(args):
    '''
        Function recieves args list of parameters.
        The function simply call the pipeline function, but called in parallel by several workers.
    '''
    df, config = args[0], args[1]
    cur_res = adding_features(df, config)
    return cur_res

def mp_handler(data, config):
    n_workers = config['n_workers']
    chunks = np.array_split(data, n_workers)
    args_list = [(chunk, config) for chunk in chunks]
    # Create a pool of n_workers processes
    pool = mp.Pool(n_workers)
    # Process in parallel
    results = pool.map(adding_features_mp, args_list)
    # Save as dictionary
    return pd.concat(results, axis=0)