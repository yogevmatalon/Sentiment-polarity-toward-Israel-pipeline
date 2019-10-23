
# coding: utf-8

# ## About this code
# 
# This script describes a pipeline for classification algorithm, using config object
# Input: a tweet or batch of tweets
# Process: apply feature engineering, modeling and classification (-1,0 or 1)
# 1. Data convertions (time field, name changing)
# 2. Adding features based on activity, tweets content and more
# 3. EDA
# 4. Supply prediction


# Imports
import pandas as pd
import numpy as np
import time
from tqdm import tqdm, tqdm_notebook
from datetime import datetime
import multiprocessing as mp
import calendar
import network_features

# Avoid trimming text in jupyter preview
pd.set_option('display.max_colwidth', -1)

# Set start time of the script
start = time.time()
tqdm.pandas(tqdm_notebook())

# #### Import thesis modules
import domain_entities
import nlp_url_features
import tweet_pre_proccess
import nlp_features
import eda
import ml_model
import slang
import cencus_data
import time_and_event
import user_bio
import sys
import os

# ## Define a pipeline

def pre_proccess(df, config = {}):
    ''' 
        Function receives a dataframe of new tweets and apply the pre-process 
        Slang convertion, Hashtags sign strip (convert to a word), tokenization, 
        Spell correction, Stop-words removal, Stemming.
    '''    
    # Get domain terms for pre-proccessing
    domain_terms = domain_entities.get_all_domain_terms()
    
    # Apply stemming, stop-words removal, slang correction and tokenization
    return tweet_pre_proccess.proccess_tweet(df, domain_terms, config.get('col', 'text'), config.get('slang', False), config.get('spell_correction', False))

def print_seperator(times = 1):
    for i in range(times):
        print('############################################################################')

def add_nlp_features(df, config):
    print('START: NLP features')
    print(' Raw NLP features')
    # RAW NLP features #
    # 1. Words type - Nouns, verbs...
    # 2. Number of capital letters, capital words
    # 3. Number of words of length = 1
    # 4. Simple features - length, #tokens, #tokens-text_length ratio
    df = nlp_features.raw_nlp_features(df,config) if config['nlp_raw'] else df

    # Hashtags and mentions #
    if config['hashtags_and_mentions']:
        print('\n Hashtags & Mentions')
        # 1. Number of hashtags in the tweet
        # 2. Number of mentions in the tweet
        # 3. Dummy feature for each important hashtag. Important hashtags declared in the hashtags_features function
        df = nlp_features.num_hashtags(df)
        df = nlp_features.num_mentions(df)
        df = nlp_features.hashtags_features(df)

    # Sentiment #
    # Add the sentiment feature (open source package)
    if config['sentiment']:
        print('NLTK Sentiment analysis')
        df['tweet_sentiment'] = df.text.progress_apply(lambda x: nlp_features.analyze_sentiment(x))

    # Dominant keywords #
    #   1. Find the k (deafult = 10) most frequent keywords per label and create dummy features
    #   2. Multiply normalized keywords' weights by the tweet sentiment and create new features
    if config.get('dominant_keywords', True):
        print('\n Dominant keywords')
        df, dominant_words = nlp_features.dominant_keywords(df, config)

    # Emotion #
    # Calculated by IBM's Watson API
    if config['emotion']:
        print(' IBM Emotion extraction')
        df = nlp_features.emotion_extraction(df, dominant_words)

    if config.get('slang', False):
        # Slang feature - number of slang words in the text
        df['num_slang_words'] = df.text.apply(lambda x: slang.slang_counter(x))

    # URL #
    #  1. Country of url determined by the domain extension
    #  2. Website domain 
    #  3. Website support bias - by country and by history (average on tweets with this domain) - TBD
    df = nlp_features.url_features(df) if config['url_features'] else df

    # Entities #
    if config['entities_features']:
        print('\n START: Entities features')
        # 1. 'Exists in tweet' features
        ## Create an iterator for features creation using function from domain_entities.py
        entities_types = {'has_pro_org': domain_entities.get_pro_orgs, 'has_anti_org': domain_entities.get_anti_orgs,
                            'has_israeli_politician': domain_entities.israeli_politics, 'has_us_politician': domain_entities.us_diplomats,
                            'has_other_politician': domain_entities.other_diplomats, 'has_terror_leaders': domain_entities.terror_leaders,
                             'has_terror_orgs': domain_entities.terror_orgs, 'has_hostile_countries': domain_entities.hostile_countries,
                            'has_news_company': domain_entities.news_companies
                          }
        # Execute all functions
        for name, entities_func in entities_types.items():
            df[name] = df.text.apply(lambda x: domain_entities.entitiy_in_tweet(x, entities_func()))

        # Adding two "summarized" columns of entities, by split the entities to pro and anti Israel
        df['hostile_entities'] = df[['has_terror_orgs','has_anti_org','has_hostile_countries','has_terror_leaders']].sum(axis=1)
        df['positive_entities'] = df[['has_israeli_politician', 'has_pro_org', 'has_us_politician']].sum(axis=1)

        # 2. Number of domain keywords in the tweet using domain_entities.get_domain_keywords
        domain_terms = domain_entities.get_all_domain_terms()
        df['num_domain_keywords'] = df.tokenized_text.apply(lambda x: nlp_features.num_terms_in_text(x, domain_terms))
        print(' - COMPLETED')


    # Tweet media and content #
    if config['Tweets_media_and_content']:
        funcs = {'has_image': nlp_features.has_image, 'has_video_ref': nlp_features.has_video_ref,
                 'has_link': nlp_features.has_link, 'has_RT': nlp_features.has_RT,
                'has_RT_req': nlp_features.has_RT_req, 'subjectivity': nlp_features.analyze_subjectivity}
        # Execute entitiy_in_tweet function for all entities
        for name, func in funcs.items():
            df[name] = df.text.apply(lambda x: func(x))
        print("\n Tweet's media and content features completed")

    # Countries support in tweet #
    # Return few features (min, sum, median...) of support for all countries mentioned in the tweet. 
    # Countries support is a prior for anti/pro-Israel sentiment in a tweet
    # The support is determined by global anti-semitic survey from http://global100.adl.org/about by Global 100 ADL organization
    df = nlp_features.get_country_support(df) if config['country_support'] else df

    print('\nCOMPLETED: NLP features')
    print('----------------------')
    print('----------------------')
    return df

def add_user_bio_features(df, config, users_db):
    print('\nSTART: User bio features')
    # Add user bio features (description analysis)
    users_bio_df = user_bio.user_bio_analysis(users_db)

    # Select only columns regrading bio or the key (screen name)
    relevant_cols = [col for col in users_bio_df.columns if col.startswith('bio_')]+['screen_name']
    users_bio_df = users_bio_df[relevant_cols]

    # Merge results
    users_bio_df = users_bio_df.rename(columns={'id_str': 'user_id'})
    df = df.merge(users_bio_df, on='screen_name', how='left', suffixes=('', '_'))

    # # Search for np.nan
    na_cols = [col for col in df.columns if df[df[col].isna()].shape[0]>0]
    print('Cols contains nulls:', na_cols)

    # Fill na in user bio features (happen when user has no description)
    bio_cols = [col for col in users_bio_df.columns if col.startswith('bio_')]
    df[bio_cols] = df[bio_cols].fillna(0)

    return df

def add_user_features(df, config, users_db):
    print('START: user features\n')
    #User feature which count the days the user in twitter

    users_db = users_db.drop(
        ['id_str', 'utc_offset', 'description', 'location', 'profile_image_url', 'name', 'time_zone',
         'profile_background_image_url', 'lang', 'profile_background_image_url_https'],
        axis=1)
    user_features = ['is_popular_RTed_rate', 'mentioned_rate', 'tweets_per_day', 'days_in_twitter', 'popularity_score',
                     'years_in_twitter', 'RTed_rate', 'likes_per_tweet', 'reply_rate', 'RT_per_tweet', 'video_ref_rate',
                     'hashtag_rate', 'mention_rate', 'link_rate', 'RT_req_rate', 'RT_prob_popularity','is_popular_score','statuses_count', 'favourites_count',
                     'followers_count', 'friends_count', 'RT_rate', 'emb_media_rate']
    # Merge results
    df = df.merge(users_db, on='screen_name', how='left', suffixes=('', '_'))

    # Convert lang_bin to boolean feature - English lan or not
    # df['lang_bin'] = df.lang_bin.apply(lambda x: True if x == 'english' else False)

    # User_time= the time the user in twitter the day he published the tweet
    df['created_at'] = df['created_at'].apply(lambda x: datetime.fromtimestamp(x))
    df['created_at_'] = df[['created_at', 'created_at_']].apply(lambda x: datetime.strptime(x['created_at_'], '%Y-%m-%d %H:%M:%S') if type(x['created_at_']) is str else x['created_at'], axis=1)
    df['time_in_twitter_unique'] = df[['created_at', 'created_at_']].apply(lambda x: (x['created_at']-x['created_at_']).days, axis=1)

    df = df.drop(['created_at_'], axis=1)
    # Convert verified to be binary: 0/1
    df['verified'] = df.verified.apply(lambda x: 0 if x == False else 1)
    df['lang_bin'] = df['lang_bin'].apply(lambda x: 1 if x == 'english' else 0)
    df['time_in_twitter_unique'] = df['time_in_twitter_unique'].fillna(0)
    df['statuses_count'] = df['statuses_count'].apply(lambda x: float(x))
    df['favourites_count'] = df['favourites_count'].apply(lambda x: float(x))
    df['followers_count'] = df['followers_count'].apply(lambda x: float(x))
    df['friends_count'] = df['friends_count'].apply(lambda x: float(x))
    df['RT_rate'] = df['RT_rate'].apply(lambda x: float(x))
    df['emb_media_rate'] = df['emb_media_rate'].apply(lambda x: float(x))

    # df[user_features] = df[user_features].fillna(0)
    print('COMPLETED: user features')

    return df


def add_time_and_event(df, config):
    # Time features
    # 1. Created_at in years since twitter foundation. Consider the growing of twitter network.
    if config['target'] not in ['relevance','support', 'subject']:
        # Time based features
        # Duration in Twitter (in years): time difference between user's registration date and Twitter foundation date.
        df['time_in_twitter_noramlized'] = df.created_at.apply(
            lambda x: (x - datetime.strptime(config['twitter_foundation_date'], '%Y-%m-%d %H:%M:%S')).total_seconds() / (
                        60 * 60 * 24 * 30 * 12))

        # 2. Time of twitting - morning/afternoon/evening/night, holiday, weekend/not weekend
        df['year'] = df['created_at'].apply(lambda x: x.year)
        df['month'] = df['created_at'].apply(lambda x: x.month)
        df['day_in_week'] = df['created_at'].apply(lambda x: calendar.day_name[x.weekday()])
        df['hour'] = df['created_at'].apply(lambda x: x.hour)
        df['time_in_day'] = df['hour'].apply(lambda x: get_time_in_day(x))
        df['is_weekend'] = df['day_in_week'].apply(
            lambda x: 1 if (x == "Saturday") or (x == 'Sunday') or (x == 'Friday') else 0)

    # Event features
    # 1. If we are in an event or not - boolean
    # 2. How many days passed since the event start time (if not finished)
    # 3. Event type - war, etc.
    df = time_and_event.get_event_features(df, config)

    return df

def get_time_in_day(time):
    if time>=5 and time<12:
        return 'Morning'
    if time>=12 and time<=15:
        return 'Noon'
    if time>=16 and time<=17:
        return 'Afternoon'
    if time>17 and time<=21:
        return 'Evening'
    if time>21 or time<=4:
        return 'Night'

def remove_features_zero_variance(df, config, categorical= False):
    '''
        Function gets df and removes features with zero variance
        categorical is a boolean variable indicates if there are categorical features in the data.
            If true, we need to manually add them to the features_filtered list
    '''
    print('- Removing features with std = 0')
    # Print columns with zero variance
    for col in [key for key, val in dict(df.std() == 0).items() if val == True]:
        print('  - {}'.format(col))

    # Keep categorical columns
    categorical_columns = ['created_at', 'name', 'screen_name','hour', 'is_weekend', 'event_name', config['target'], 'text', 'tokenized_text'] if categorical else [config['target'], 'text', 'tokenized_text']
    # Keep only columns that are in the df
    categorical_columns = [col for col in categorical_columns if col in df.columns]
    # Keep columns with std>0 and categorical columns only
    features_filtered = list([key for key, val in dict(df.std() > 0).items() if val == True] + categorical_columns)
    df = df[features_filtered]
    return df

def remove_low_correlated_features(df, config, most_correlated_variables, corr, categorical= False):
    '''
    Function gets a df and corr matrix and remove features with low correlation
    categorical is a boolean variable indicates if there are categorical features in the data.
        If true, we need to manually add them to the features_filtered list
    '''
    #most_correlated_variables
    #### Inspect correlations to the target variable
    corr_with_label = pd.DataFrame(corr[config['target']].iloc[2:])
    corr_with_label[config['target']] = corr_with_label.apply(lambda x: abs(x))
    corr_with_label.sort_values(by=[config['target']], ascending=False).head()

    print('- Removing low correlated features (with target variable)')
    # Threshold to filter low correlated features
    corr_threshold = corr_with_label[config['target']].quantile(q=config['corr_per_thresh'])
    categorical_columns = ['created_at', 'name', 'screen_name', 'hour', 'event_name', config['target'], 'text', 'tokenized_text'] if categorical else [config['target'], 'text', 'tokenized_text']
    categorical_columns = [col for col in categorical_columns if col in df.columns]

    # Get a list of features that above the threshold
    features_filtered = list(corr_with_label[corr_with_label[config['target']] > corr_threshold].index.tolist() + categorical_columns)
    print('  iltered features (corr): {}'.format([col for col in list(df.columns) if col not in features_filtered]))
    df = df[features_filtered]
    return df

def remove_correlated_features(df, config, most_correlated_variables, corr, categorical= None):
    ''' Function gets a df and remove correlated features (features do no include the target) '''
    print('Removing correlated features')
    #### Remove features that correlated to each other
    print('  - TBD')
    return df

def remove_features_by_importance(df, config, categorical= False):
    '''
    Function gets a df remove features with low importance by random forest- featue importance.
    categorical is a boolean variable indicates if there are categorical features in the data.
        If true, we need to manually add them to the features_filtered list
        '''
    unnecessary_features = [config['target'], 'text', 'tokenized_text','screen_name','name','favorite_count', 'words_types',
                                                               'retweet_count', 'id','user_id', 'created_at', 'event_name', 'time_in_day', 'day_in_week']
    features = [col for col in df.columns.tolist() if col not in unnecessary_features]
    na_cols = df[features].columns[df[features].isna().any()].tolist()
    if na_cols:
        print('Cols with NAs: {}'.format([col for col in df.column if df[df[col].isna()].shape[0]>0]))
    else:
        print(' - No NAs in the features')

    # Get features' importance
    importances_dict = eda.feature_importance(df, features, config['target'])  # dict format - feature: importance (%)

    # Remove features with low importance - under the 25 percentile (of importance score)
    # Find the threshold to filter features by importance - the percentile is determined in the config object - importance_per_thresh
    importance_threshold = np.percentile(np.array([list(importances_dict.values())]), config['importance_per_thresh'])
    print('- Removing features with low importance')

    # Important cols -> cols above threshold
    # categorical_columns = ['created_at', 'name', 'screen_name', 'hour', 'event_name',
    #                            config['target'], 'text', 'tokenized_text'] if categorical else [config['target'],'text','tokenized_text']
    # important_cols = list([feature for feature, importance in list(importances_dict.items()) if
    #                   importance >= importance_threshold] + categorical_columns)
    filtered_cols = [feature for feature, importance in list(importances_dict.items()) if
                                  importance <= importance_threshold]

    print('  Num features before filtering: {}'.format(df.shape[1]))
    df = df[[col for col in df.columns if col not in filtered_cols]]
    print('  Num features after filtering: {}'.format(df.shape[1]))
    print('  Num filtered cols:', len(filtered_cols))
    return df

def feature_selection(df, config, categorical= False):
    ## Feature selection
    print('\n- - - - - - - - - - -')
    print('START: FEATURE SELECTION')

    # Remove id column
    #df = df.drop(['id'], axis=1)

    ### Remove features with std = 0
    df = remove_features_zero_variance(df, config, categorical) if config['remove_features_zero_variance'] else df

    # Correlations
    features = df._get_numeric_data().columns
    if config['remove_correlated_features'] or config['remove_low_correlated_features']:
        most_correlated_variables, corr = eda.plot_correlation_matrix(df, features)

    ### Remove features which correlated to each other - TBD
    df = remove_correlated_features(df, config, most_correlated_variables, corr, categorical) if config['remove_correlated_features'] else df

    #### Remove features with low correlation to the target variable
    df = remove_low_correlated_features(df, config, most_correlated_variables, corr, categorical) if config['remove_low_correlated_features'] else df

    ### Feature selection using feature importance
    df = remove_features_by_importance(df, config, categorical) if config['feature_importance'] else df

    # Reset index before PCA and Model phase
    df.reset_index()

    ### Dimensionality reduction
    if config.get('PCA',False):
        print('- Applying PCA')
        # apply_PCA input: df, label_name, features, variance tresh
        pre_features = df._get_numeric_data().columns
        result = ml_model.apply_PCA(df, config['target'], pre_features, config['PCA_var'])
        df = result[0]
        pca = result[1]
        features = [col for col in df.columns.tolist() if col not in [config['target'], 'text', 'tokenized_text']]
        # Show PCA result statistical data
        df.describe()
        return (df, pca, pre_features)
    print('\nCOMPLETED: Feature Selection')
    print('----------------------\n')
    return df

def pipeline(df, config):
    '''
        Input: a dataframe of records from twitter
        Output: classification anti/pro-Israel or neutral per record
    '''
    print('----------------------')
    df = pre_proccess(df, config) if config.get('pre_proccess',True) else df

    pre_proccess_time = time.time()

    ## Features
    # Add NLP feautures
    df = add_nlp_features(df, config) if config.get('nlp_features',True) else df

    # Add user features and user bio features
    if config.get('user_features', False) or config.get('user_bio', False):
        # Load users db in order to import user metadata and features
        users_db = pd.read_csv('../../data/users_DB_no_location.csv')
        # Remove unwanted features
        users_db = users_db.iloc[:, 2:]
        users_db = users_db.drop(users_db[users_db.id_str.str[0] == 'h'].index, axis=0)
        users_db = users_db.dropna(subset=['id_str'], axis=0)
        # Remove duplicates
        users_db = users_db.drop_duplicates(subset=['id_str'], keep='first')

    # Add user metadata (users_db) features
    df = add_user_features(df, config, users_db) if config.get('user_features',False) else df

    # Add network features
    df = network_features.add_network_features(df, config.get('load_network_data', True), path='../../data/network_data/', target=config['target']) if config.get('network_features', False) else df

    # Add user bio features
    df = add_user_bio_features(df, config, users_db) if config.get('user_bio',False) else df

    # Add time and event features
    df = add_time_and_event(df, config) if config.get('time_and_event', False) else df

    # Merge completed -> drop 'screen_name' feature
    # df = df.drop(['screen_name'], axis=1) if 'screen_name' in df.columns else df

    if config['learning']: df = df.drop(['id'], axis=1)

    print('Feature engineering COMPLETED')
    # Apply feature selection
    if config.get('PCA',False):
        if config['learning']:
            result = feature_selection(df, config, True) if config.get('feature_selection', False) else df
            return result[0], result[1], result[2]
        else:
            return df

    # Apply feature selection
    df = feature_selection(df, config, True) if config.get('feature_selection', False) else df

    print('Pipeline COMPLETED')

    return df

def pipeline_mp(args):
    '''
        Function recieves args list of parameters.
        The function simply call the pipeline function, but called in parallel by several workers.
    '''
    df, config = args[0], args[1]
    cur_res = pipeline(df, config)
    return cur_res

def mp_handler(data, config):
    n_workers = config['n_workers']
    chunks = np.array_split(data, n_workers)
    args_list = [(chunk, config) for chunk in chunks]

    # Create a pool of n_workers processes
    pool = mp.Pool(n_workers)
    # Process in parallel
    results = pool.map(pipeline_mp, args_list)
    return pd.concat(results, axis=0)

def convert_to_dummy(df):
    '''
    Function recieves df and convert categorical features to dummy features.
    Relevant only to the virality algorithm.
    '''
    time_in_day_dummy_df = pd.get_dummies(df['time_in_day'])
    day_in_week_dummy_df = pd.get_dummies(df['day_in_week'])
    support_dummy_df = pd.get_dummies(df['support'])
    #event_type_dummy_df = pd.get_dummies(df['event_type'])
    df = pd.concat([time_in_day_dummy_df, day_in_week_dummy_df, support_dummy_df, df], axis=1)
    df.drop(['time_in_day', 'day_in_week', 'support'], axis=1, inplace=True)
    return df

