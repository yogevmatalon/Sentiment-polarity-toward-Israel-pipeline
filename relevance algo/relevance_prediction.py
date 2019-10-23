'''
This script apply the relevance classification algorithm to new unseen and unlabeled data.
Input: a tweet or batch of tweets
Process: apply pre-process, feature engineering, and classification (Not relevant (2) vs all
Output: a classification per input: irrelevant or relevant
1. Using the Pipeline.py for pre-process, feature engineering (NLP, entities, user features) and feature selection
    Dominant words list is taken from learning
    Features are eventually filtered according to the final features in learning
2. Saving irrelevant data
3. Output relevant data
'''

#Imports packages
import pandas as pd
import numpy as np
import os
import pickle
import sys

#Imports modules
os.chdir('../pipeline')
import pipeline

def relevance_score(tweet, predict_prob=True, config=None):
    '''
    :param tweet: a string to be analyzed
    :param config: configuration dictionary
    :param predict_prob: True if we want to get a probability, False if only classification
    :return:
    '''
    # Function gets a single string (tweet) and config object and return the string with relevance prediction
    os.chdir('../relevance algo')
    # Taking only needed features
    with open('relevance_features_list.pickle', 'rb') as handle:
        train_features = pickle.load(handle)

    emotions = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Analytical", "Confident", "Tentative", "Openness",
                "Conscientiousness", "Extraversion", "Agreeableness", "Emotional Range"]
    to_extract_emotions = False
    for emo in emotions:
        if emo in train_features:
            to_extract_emotions=True

    if not config:
        config = {
            'learning': False, 'target': 'relevance','load_df_from_pickle': False, 'slang': True, 'spell_correction': False, 'col': 'text',
            'nlp_features': True,'dominant_keywords': True,'user_features': False,'time_and_event': False,'network_features': False,
            'load_network_data': False,'nlp_raw': True,
            'sentiment': True if 'tweet_sentiment' in train_features else False,
            'emotion': to_extract_emotions,
            'word_type': False,
            'hashtags_and_mentions': True,'url_features': False,'Tweets_media_and_content': True,'country_support': False,'entities_features': True
        }

    # Features creation - Run pipeline
    data = pd.DataFrame(data=[tweet], columns=['text'])
    df = pipeline.pipeline(data.copy(), config)

    # Filter df features
    df = df[train_features]

    ## Loading model
    with open('relevance_model.pickle', 'rb') as handle:
        model = pickle.load(handle)

    # Predict label
    if predict_prob:
        # Probability of being not(!) relevant
        df['relevance'] = model[2].predict_proba(df)[:, 1]
    else:
        df['relevance'] = int(model[2].predict(df))

    return df['relevance'][0]


def predict(data, config):
    # Function gets a data-frame and config object and return the data with relevance prediction
    os.chdir('../relevance algo')
    # Features creation - Run pipeline
    print('Creating features for relevance algo')
    if config['distributed_pipeline'] and not config['load_df_from_pickle']:
        print(' - Distributed to {} workers'.format(config['n_workers']))
        df = pipeline.mp_handler(data.copy(), config)
    else:
        df = pipeline.pipeline(data.copy(), config) # if not config['load_df_from_pickle'] else pd.read_csv('../pipeline/predictions/df_relevance_no_filter_no_prediction.tsv', header=0, sep='\t')
        print('Pipeline results loaded from tsv')

    df_full_features = df.copy()

    print(' - COMPLETED')

    # Taking only needed features
    with open('relevance_features_list.pickle', 'rb') as handle:
        train_features = pickle.load(handle)

    # Train_features.append('id')
    # print('train features: \n', pd.DataFrame(train_features, columns=['features']))
    # Filter df features
    df = df[train_features]

    # Export a df with the relevant features for the relevance algo (only)
    # df.to_csv('../pipeline/predictions/data_w_relevance_features_no_pred.tsv', header=True, sep='\t')
    # print('Exported: data_w_relevance_features_no_pred.tsv')

    # Saving df
    # if not config['load_df_from_pickle']:
    #     # df_full_features is a df with all features created for relevance algorithm, without feature selection
    #     df_full_features.to_csv('../pipeline/predictions/df_relevance_no_filter_no_prediction.tsv', header=True, sep='\t')
    #     print('Exported: df_relevance_no_filter_no_prediction.tsv')

    # Predict
        ## Loading model
    with open('../../data/classifiers/relevance_model.pickle', 'rb') as handle:
        model = pickle.load(handle)

    # Predict label
    print('Predicting relevance')
    df['relevance'] = model[2].predict(df)
    print(' - COMPLETED')

    # Change the relevance label of 'not relevant' tweets to be 2
    df['relevance'] = df['relevance'].apply(lambda x: x if x!=1 else 2)
    print('COMPLETED: Relevance prediction')
    return df, df_full_features

def remove_irrelevants(df, full_df, file_prefix='', file_suffix=''):
    print('Removing irrelevants tweets')
    # Function gets:
    # 1. df - only relevance algo features (filtered) + relevance prediction
    # 2. full_df - all features (no filtering) but without relevance prediction
    full_df = pd.concat([full_df, df[['relevance']]], axis=1)

    # Save all irrelevant tweets (for debug, evaluation). Export to pickle
    # full_df[full_df['relevance'] == 2].to_pickle('irrelevant_df.pkl')
    full_df = full_df[full_df['relevance'] != 2]
    print(' - COMPLETED')

    # Export to tsv
    # full_df.to_csv('../../data/predictions/{}relevance_prediction{}.tsv'.format(file_prefix, file_suffix), header=True, sep='\t')
    print('# of relevant tweets: {}. {} tweets have been removed'.format(full_df.shape[0], df.shape[0]-full_df.shape[0]))
    return full_df

def add_relevance(df, data):
    print('Adding relevance feature')
    # Merge relevance column with original data
    return pd.concat([data, df[['relevance']]], axis=1)