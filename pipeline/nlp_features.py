
# coding: utf-8


import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from textblob import TextBlob
from scipy.stats import entropy
import collections
from collections import Counter
import ast
import pickle
import multiprocessing as mp

# Our packages
import nlp_url_features
import country_support
import eda

def get_negation_words():
    return {'not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                      'even','though', 'yet', "n't", 'no', 'none', 'noone', 'nobody', 'nothing', 'nowhere',
                      'never', 'hardly', 'scarcely', 'barely'}

def raw_nlp_features(df,config):
    # Function get a dataframe and apply 3 different features to its 'text' column:
    # 1. Words type - Nouns, verbs...
    if config['word_type']:
        df= creating_word_type(df)
        print('  - Word classes features completed (1/5)')
    else:
        print('  - Word classes features doesnt need to be done (1/5)')

    # 2. Number of capital letters, capital words
    df['num_capital_letters'] = df.text.apply(lambda x: sum([1 if char.isupper() else 0 for char in x]))
    df['num_capital_words'] = df.text.apply(lambda x: sum([1 if word.isupper() else 0 for word in x.split()]))
    print('  - Capital letters features completed (2/5)')

    # 3. Number of words of length = 1
    df['num_words_len_1'] = df.text.apply(lambda x: sum([1 if len(word) ==1 else 0 for word in x.split()]))
    print('  - 1 length words features completed (3/5)')

    # 4. Simple features - length, #tokens, #tokens-text_length ratio
    df['tweet_length'] = df.text.apply(lambda x: len(x))
    # Num of tokens is at least 1, even if list of tokens is empty - both for semantic reasons
    # and due to mathematical constrain - text_length_vs_tokens_ratio should have denominator > 0
    df['num_tokens'] = df.tokenized_text.apply(lambda x: max(1,len(x)))
    df['text_length_vs_tokens_ratio'] = df.tweet_length / df.num_tokens
    print('  - Simple features completed (4/5)')

    negation_words = get_negation_words()
    df['num_negation_words'] = df.tokenized_text.progress_apply(lambda x: sum([1 if term in negation_words else 0 for term in x]))
    print('  - Negation words feature completed (5/5)')
    return df

def num_mentions(df):
    # The function counts the number of mentions (words starts with @) in the tweet
    df['num_mentions'] = df.text.apply(lambda x: sum(1 if term.startswith('@') else 0 for term in x.split()))
    print('  - Mentions feature completed')
    return df

def calculate_maention_rate(df, config):
    df['mentioned_rate'], df['followers_mention_user'] = df.text.apply(lambda x: user_mention_rate(x))
    return df

def user_mention_rate(tweet):
    '''
    Function recieves df and for each tweet fill the user mention rate by users df.
    if no user mentioned in tweet - fill it 0.
    '''
    users_db = pd.read_csv('../../data/users_DB_no_location.csv')
    user_mentioned = [term[1:] for term in tweet.split() if term.startswith('@')] #get the users mentioned in tweet
    if len(user_mentioned)==0:
        mentioned_rate = 0
        followers_mention_user = 0
    else:
        mentioned_rate = 0
        followers_mention_user = 0
        for user in user_mentioned:
            mentioned_rate+= users_db[users_db.screen_name==user].mentioned_rate
            followers_mention_user += users_db[users_db.screen_name==user].followers_count
        mentioned_rate = mentioned_rate/len(user_mentioned)
        followers_mention_user = followers_mention_user/len(user_mentioned)
    return mentioned_rate, followers_mention_user

def num_hashtags(df):
    # The function counts the number of hashtags (words starts with @) in the tweet
    df['num_hashtags'] = df.text.apply(lambda x: sum(1 if term.startswith('#') else 0 for term in x.split()))
    return df


def hashtags_features(df):
    # Functions gets a df and return dummy feature for each hashtag from the list (per tweet)
    # Important hashtags are generated from the entire corpus -> taking the most frequent
    hashtags = {'israel', 'palestine', 'sjp', 'gaza', 'bds', 'lovely', 'apartheid', 'flotilla', 'gazaunderattack',
                'tcot', 'news', 'land', 'occupation', 'standwithisrael', 'proisrael', 'israstine', 'standwithus',
                'israeli', 'iran', 'freepalestine', 'health', 'syria', 'hamas', 'egypt', 'earthquake',
                'palestinians', 'jerusalem', 'istandwithisrael', 'solidaritywavebds', 'apartheidisrael',
                'boycottisrael', 'politics', 'us', 'usa', 'nufc', 'palestinian', 'cnn', 'un', 'israels', 'jewish',
                'prayforgaza', 'zionism', 'obama', 'idf', 'humanrights', 'turkey', 'economy', 'icc4israel',
                'letssavegaza', 'abc', 'occupation', 'jews', 'lebanon', 'nbc', 'westbank', 'freedomflotilla',
                'peace', 'netanyahu', 'middleeast', 'stopincitement', 'prayforisrael', 'supportisrael', 'supportidf',
                'antisemitism', 'islam', 'freegaza', 'greatreturnmarch', 'boycotteurovision', 'israelunderattack',
                'israelilivesmatter', 'istandwithisrael', 'prayforpalestine', 'israelforever', 'wesupportisrael',
                'jerusalemintifada', 'jewishlivesmatter', 'israeliapartheidweek'}
    for hashtag in hashtags:
        # Binary feature - if the hashtag is in the tweet or not
        df['hash_#{}'.format(hashtag)] = df.text.apply(lambda x: 1 if str('#' + hashtag.lower()) in x.lower() else 0)

    # Number of total important hashtags in the tweet
    hash_cols = [col for col in df.columns if col.startswith('hash_')]
    df['total_important_hashtags'] = df[hash_cols].sum(axis=1)
    print('  - Hashtags features completed')
    return df


def match_couple_in_text(tokenized_text, word1, word2):
    '''
    :param tokenized_text: tokenized tweet
    :param word1, word2: 2 words to be searched in the tweet. They can be a string, or group of strings (only one of them)
    :return: 1 if word2 appear after word1, else 0
    '''
    if type(word1) == 'str' and type(word2) == 'str':
        # If both words are in the text
        if word1 in tokenized_text and word2 in tokenized_text:
            return 1 if tokenized_text.index(word2) > tokenized_text.index(word1) else 0
        else:
            return 0

    elif type(word1) != 'str':
        # word1 is the group
        num_of_group_in_text = sum([1 if word in tokenized_text else 0 for word in word1])

        if num_of_group_in_text in [None, np.nan]:
            print(
            '!! Text: {},\n  word1: {}, word2: {}, \nnum_of_group_in_text: {}'.format(tokenized_text, word1, word2,
                                                                                      num_of_group_in_text))

        # If both words are in the text
        if word2 in tokenized_text and num_of_group_in_text > 0:
            # Find the index of the first word from the group in the text
            first_of_group_in_text = min([tokenized_text.index(word) for word in word1 if word in tokenized_text])
            res = 1 if tokenized_text.index(word2) > first_of_group_in_text else 0

            if res in [None, np.nan]:
                print(
                '!! Text: {},\n  word1: {}, word2: {}, \nnum_of_group_in_text: {}'.format(tokenized_text, word1, word2,
                                                                                          res))

            return res if res is not np.nan else 0
    else:
        # word2 is the group
        num_of_group_in_text = sum([1 if word in tokenized_text else 0 for word in word2])
        # If both words are in the text
        if word1 in tokenized_text and num_of_group_in_text > 0:
            first_of_group_in_text = min([tokenized_text.index(word) for word in word2 if word in tokenized_text])
            res = 1 if first_of_group_in_text > tokenized_text.index(word1) else 0

            if res in [None, np.nan]:
                print(
                '!! Text: {},\n  word1: {}, word2: {}, \nnum_of_group_in_text: {}'.format(tokenized_text, word1, word2,
                                                                                          res))

            return res if res is not np.nan else 0
    return 0


def create_couples_features(df, dominant_words_norm, min_appearances=4):
    '''
    :param df: data frame to add the features to
    :param dominant_words_norm: set of tuples - (dominant_word, sqrt(idf))
    :return: The df with new features. Each feature contains 2 dominant words: (word1,word2).
            The feature's value will be 1 if word2 appear after word1, else 0 (also if one of them isn't in the tweet).
            Another set of features: couples of strong words and negation_word (any negation word)
    '''
    for word1 in dominant_words_norm:  # word1: ( word, sqrt(idf) )
        for word2 in dominant_words_norm:
            if word1[0] != word2[0]:
                df['couple:{}->{}'.format(word1[0], word2[0])] = df.tokenized_text.apply(
                    lambda x: match_couple_in_text(x, word1[0], word2[0]))
                # Check that the couple appear in the data at least min_appearances times. If not - delete this feature
                if df['couple:{}->{}'.format(word1[0], word2[0])].sum() < min_appearances:
                    df = df.drop(columns=['couple:{}->{}'.format(word1[0], word2[0])])

    # Couples of strong words and negation words
    print(' - dominant words + negation words')
    neg_words = get_negation_words()
    for word in dominant_words_norm:
        df['couple:{}->NegationWord'.format(word[0])] = df.tokenized_text.apply(
            lambda x: match_couple_in_text(x, word[0], neg_words))
        df['couple:NegationWord->{}'.format(word[0])] = df.tokenized_text.apply(
            lambda x: match_couple_in_text(x, neg_words, word[0]))

        # Check that the couple appear in the data at least min_appearances times. If not - delete this feature
        if df['couple:{}->NegationWord'.format(word[0])].sum() < min_appearances:
            df = df.drop(columns=['couple:{}->NegationWord'.format(word[0])])
        if df['couple:NegationWord->{}'.format(word[0])].sum() < min_appearances:
            df = df.drop(columns=['couple:NegationWord->{}'.format(word[0])])

    return df


def add_ready_words_columns(df, dominant_tokens):
    '''
    Function recieves a data frame and builds a data frame of bag of words count
    Each tweet has tokenized_text, which is a list of all tokens in the tweet (after pre-process: stemming, etc...)
    We count each token, for each tweet, and then construct a data-frame.
    :param df: data frame
    :param dominant_tokens: ready group of dominant_tokens to use (!) - this is the difference between add_ready_words_columns and add_words_columns
    '''

    # Copy and reset index in order to iterate the df properly
    df.reset_index(drop=True)
    print(' - Creating Bag-of-Words features')

    # Build a dict that will save counters for each word
    all_keywords = {}
    for i, row in df.iterrows():
        all_keywords[i] = {}
        for word in dominant_tokens:
            all_keywords[i]['word_{}'.format(word)] = 1 if word in row.tokenized_text else 0

    # Build a df from the dict
    keywords_df = pd.DataFrame.from_dict(all_keywords, orient='index')
    # NaN means the words is not in the df -> set as zero
    keywords_df = keywords_df.fillna(value=0)
    # Merge all words columns with the input df
    df = pd.merge(df, keywords_df, left_index=True, right_index=True, suffixes=['', '_'])
    # Save words columns names
    return df


def add_words_columns(df, config):
    '''
    Function recieves a data frame and builds a data frame of bag of words count
    Each tweet has tokenized_text, which is a list of all tokens in the tweet (after pre-process: stemming, etc...)
    We count each token, for each tweet, and then construct a data-frame.
    :param df: data frame
    :param config: Used to check the algorithm target. Can be sent {target: 'target_name'} only.
    :return: The input_df with column for each word in text_column, and list of features created: features.
             Each word column starts with 'word_'
    '''
    print(' - Creating bag-of-words features')
    # Copy and reset index in order to iterate the df properly]
    # Build a dict that will save counters for each word
    if config.get('all_words', True):
        all_keywords = {}
        for i, row in df.iterrows():
            text_list = row.tokenized_text
            if len(text_list) > 0:
                all_keywords[i] = {}
                for word in text_list:
                    all_keywords[i]['word_{}'.format(word)] = 1  # all_keywords[i].get(word,0)+1
    else:
        all_keywords = add_most_common_words(df)

    # Build a df from the dict
    keywords_df = pd.DataFrame.from_dict(all_keywords, orient='index')
    # NaN means the words is not in the df -> set as zero
    keywords_df = keywords_df.fillna(value=0)

    # Sort keywords_df for fixing columns order
    keywords_df = keywords_df[sorted(keywords_df.columns)]

    # Merge all words columns with the input df
    df = pd.merge(df, keywords_df, left_index=True, right_index=True, suffixes=['', '_'])
    # Save words columns names
    features = [col for col in keywords_df.columns if col.startswith('word_')]
    print('  - COMPLETED: {} columns created'.format(len(features)))
    return df, features

def add_most_common_words(df, k=500):
    '''
        Function recieves a data frame and builds a dictionary of bag of the most common words count
        Each tweet has tokenized_text, which is a list of all tokens in the tweet (after pre-process: stemming, etc...)
        We count each token, for each tweet, and then construct a dictionary.
        :param df: data frame
        :return: The most common words in tweets
        '''
    x = sum(df.tokenized_text, [])
    c = Counter(x)
    common_words = c.most_common(k)
    all_keywords = {}
    for i, row in df.iterrows():
        text_list = row.tokenized_text
        if len(text_list) > 0:
            all_keywords[i] = {}
            for word in common_words:
                if word[0] in text_list:
                    all_keywords[i]['word_{}'.format(word[0])] = 1  # all_keywords[i].get(word,0)+1
    return all_keywords

def calc_adj_error_score(pk):
    # pk - list of probabilities. Function return the max adjusted error score of the distribution.
    # In case that the max probability is neutral, each possible error is evenly important (neutral-positive/ neutral-negative)
    # In case that the max probability is positive/negative, neutral error is 0.5 error, while the contrast is 1 error.
    if max(pk) == pk[1]:  # Max prob = neutral
        # Error
        return 1 * (1 - pk[1])
    elif max(pk) == pk[2]:  # Max prob = positive
        return 1 - 0.5 * pk[1] - pk[0]
    elif max(pk) == pk[0]:  # Max prob = negative
        return 1 - 0.5 * pk[1] - pk[2]
    return None


def calc_words_metric(df, config, words_columns, metric='entropy'):
    # Function gets a data-frame calculates a metric (e.g. entropy) of its words-based columns
    # The function return sorted list of [feature, metric]
    metric_mapper = {'entropy': {'func': entropy, 'target': 'minimize'},
                     'purity': {'func': max, 'target': 'maximize'},
                     'smart_error': {'func': calc_adj_error_score, 'target': 'minimize'}
                     }
    res = {}
    # Iterate over all columns (i.e. words)
    num_rows = df.shape[0]
    df_target = pd.DataFrame(df[config['target']].unique(), columns=[config['target']]).dropna()
    for col in words_columns:
        # Calc pk - distribution of each class, for each column (i.e. word)
        df_word_exist = df[[col, config['target']]].loc[df[col] == 1]
        df_word_not_exist = df[[col, config['target']]].loc[df[col] == 0]
        pk1 = df_target.merge(df_word_exist.groupby(config['target']).agg({col: np.size}).reset_index(),
                              left_on=config['target'], right_on=config['target'], how='left').fillna(0)[col].tolist()
        pk2 = df_target.merge(df_word_not_exist.groupby(config['target']).agg({col: np.size}).reset_index(),
                              left_on=config['target'], right_on=config['target'], how='left').fillna(0)[col].tolist()

        # Save sizes for averaging
        pk1_size, pk2_size = len(df_word_exist), len(df_word_not_exist)
        # Convert pk to prob
        pk1, pk2 = [p / pk1_size for p in pk1], [p / pk2_size for p in pk2]

        # Calc the metric of pk, using the input metric and the metric_mapper, averages using the sizes of the split
        res[col] = (metric_mapper[metric]['func'](pk1) * pk1_size + metric_mapper[metric]['func'](
            pk2) * pk2_size) / num_rows

    res_items = list(res.items())
    res_items.sort(key=lambda x: x[1], reverse=True if metric_mapper[metric]['target'] == 'maximize' else False)
    return res_items


def get_dominant_keywords(df, config):
    # Function included in dominant_keywords function. This function is for extracting the dominant words for production phase (or debuging).
    # k: number of dominant keywords to return for each label
    k = config.get('num_dominant_words', 10)

    if config['learning'] and not config.get('subject', False):
        # If this is the train set then find the dominant words.
        # Find the k (=10 by default) most frequent keywords per label and create dummy features
        df_cols = list(df.columns)
        df, words_columns = add_words_columns(df, config)

        # Filter words with less than 4 appearances
        n_prev = len(words_columns)
        words_columns = [word_col for word_col in words_columns if
                         df[word_col].sum() >= config.get('min_couple_appearances', 0)]

        print('Words frequency filter: {} word features removed'.format(len(words_columns) - n_prev))

        # Filter using importance (metric)
        words_importance = calc_words_metric(df, config, words_columns,
                                             config.get('dominant_keywords_metric', 'entropy'))

        # Take the k most important words
        dominant_tokens = words_importance[:k]

        # Add this columns to final columns list
        df_cols.extend([col[0] for col in dominant_tokens])
        # Apply filtering
        df = df[df_cols]

        # Sort columms of df in a dictionary sort
        df = df[sorted(df.columns)]

        # Save algorithm dominant words
        with open('../{} algo/{}_dominant_words.pickle'.format(config['target'], config['target']), 'wb') as handle:
            pickle.dump(dominant_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif config.get('subject', False):
        # Subject algo uses the relevance algo dominant words
        with open('../relevance algo/relevance_dominant_words.pickle', 'rb') as handle:
            dominant_tokens = pickle.load(handle)
        dominant_words = [feature[0].split('_')[1] for feature in dominant_tokens if feature[0].startswith('word_')]
        df = add_ready_words_columns(df, dominant_words)

    else:  # Use dominant words using the features_list of the model
        print(' - loading dominant words')
        with open('../{} algo/{}_features_list.pickle'.format(config['target'], config['target']), 'rb') as handle:
            dominant_tokens = pickle.load(handle)

        # We get all model's features. We need to extract the dominant words from them:
        dominant_words = [feature.split('_')[1] for feature in dominant_tokens if feature.startswith('word_')]
        df = add_ready_words_columns(df, dominant_words)

    return df, dominant_tokens


def dominant_keywords(df, config):
    '''
    :param df: a dataframes contains tweets, tokenized in the tokenized_text column
    :param config: dictionary of configurations
    :return: modified df
    Action:
        1. Find the k (default = 10) most frequent keywords per label and create dummy features
        2. Multiply normalized keywords' weights by the tweet sentiment and create new features
    '''
    # Add dominant words features
    # In case we are in the learning phase - calculate it. On production - import dominant_words.pickle
    df, dominant_words = get_dominant_keywords(df, config)

    # Add the new words features
    for token in dominant_words:
        # token: (word, metric value)
        # Number of appearances for each word in the text
        # When learning - dominant_words is list of tuples. When predicting - list of strings
        if type(token) is str:
            df['{}_count'.format(token)] = df.tokenized_text.apply(lambda x: x.count(token))
        else:
            df['{}_count'.format(token[0])] = df.tokenized_text.apply(lambda x: x.count(token[0]))

    # Number of dominant_words in total
    print(' - num_dominant_words')
    df['num_dominant_words'] = df.tokenized_text.apply(lambda x: sum([x.count(token[0]) for token in dominant_words]))

    # Create feature based on couple of words
    if config['learning'] == True:
        min_appearances = config.get('min_word_appearances', 4)
    else:
        min_appearances = 0

    # print(' - Dominant couples')
    # df = create_couples_features(df, dominant_words, min_appearances) # We turned it off since no couple survived the feature selection

    print(' - Sentiment * dominant words')
    if config['sentiment']:
        # Sentiment * token_weight
        for token in dominant_words:
            # When learning - dominant_words is list of tuples. When predicting - list of strings
            if type(token) is str:
                df['sentiment * {}_count'.format(token)] = df['tweet_sentiment'] * df['{}_count'.format(token)]
            else:
                df['sentiment * {}_count'.format(token[0])] = df['tweet_sentiment'] * df['{}_count'.format(token[0])]

    print('COMPLETED: Dominant words')
    return df, dominant_words


def remove_dominant_words(df, config):
    '''
        The functions recieves a df and remove all dominant_words features from it.
        Used in online pipeline when switching between targets (and therefore the dominant words list is different)
    '''
    # Get all features need to be removed
    # features_mask = [col for col in df.columns if (col.startswith('word_')
    #                                                or col.startswith('couple:')
    #                                                or col.startswith('sentiment *')
    #                                                or col == 'dominant_words_total'
    #                                                or col == 'dominant_words_total_weighted'
    #                                                )]
    # # Remove features
    # for feature in features_mask:
    #     if feature in df.columns:
    #         df.drop(feature, axis=1, inplace=True)

    df = df[[col for col in df.columns if not (col.startswith('word_')
                                                   or col.startswith('couple:')
                                                   or col.startswith('sentiment *')
                                                   or col == 'dominant_words_total'
                                                   or col == 'dominant_words_total_weighted'
                                                   ) ] ]
    return df


def has_image(text):
    text = text.lower()
    if 'pic.twitter.com' in text:
        return 1
    else:
        return 0


def has_video_ref(text):
    '''checks if a tweet contains a link to a video. One of 3 options:
            1. Direct link to video (youtube, vimeo, vine...)
            2. Bitly address AND act word (like 'watch')
            3. '/Video/' or '.MP4' keywords'''
    text = text.lower()
    if (('youtube' in text) or ('instagram.com' in text) or ('vimeo.com' in text) or ('youtu.' in text) or (
        'vine.' in text) or ('.mp4' in text) or ('/video/' in text) or
            ((('bitly.com' in text) or ('bit.ly' in text) or ('j.mp' in text) or ('t.co' in text)) and (
                    ('video' in text) or ('watch' in text) or ('see' in text)))):
        return 1
    else:
        return 0


def has_link(text):
    text = text.lower()
    if ('http:' in text) or ('https:' in text) or ('www.' in text):
        return 1
    return 0


def has_RT(text):
    '''  '''
    if ('RT @' in text) or (text.startswith('RT')):
        return 1
    else:
        return 0


def has_RT_req(text):
    ''' If the user asked people to RT his tweet '''
    text = text.lower()
    if (('please rt' in text) or ('rt please' in text) or ('please share' in text) or ('share please' in text) or
            ('share pls' in text) or ('pls rt' in text) or ('rt pls' in text) or ('share this' in text) or (
        'rt this' in text)):
        return 1
    else:
        return 0


def analyze_sentiment(tweet):
    '''Utility function to classify the polarity of a tweet using textblob'''
    tweet = tweet.replace('#', '')
    tweet = tweet.replace('-', ' ')
    negative_words = ['genocide','apartheid','warcrimes', 'occupier','occupied', 'occupation','boycott', 'freepalestine','jewhate','suppression',
                      'oppression', 'settler', 'settlers','settlements', 'settlement', 'boycottisrael','justice','infiltrated',
                      'justice4palestine','stopisraeliapartheid','robbers','stole','stoparmingsrael','preach','sanction','sanctions']
    positive_words = ['loveisrael', 'supportisrael', 'rockets', 'rocket','innovation','startupnation','support','aid']
    sid = SIA()
    for word in negative_words:
        if word=='genocide' or word=='apartheid':
            sid.lexicon[word] = -2
        else:
            sid.lexicon[word] = -1
    for word in positive_words:
        sid.lexicon[word] = 1
    ss = sid.polarity_scores(tweet.lower())
    # taking the polarity
    return ss['compound']


def analyze_subjectivity(tweet):
    '''
        Objectivity - subjectivity score (float)
        1 = subjective, 0 = objective
    '''
    return TextBlob(tweet).sentiment[1]


def analyze_sentiment_TextBlob(tweet):
    '''
        Polarity/sentiment score (float)
        1 = positive, -1 = negative
    '''
    return TextBlob(tweet).sentiment.polarity


def url_features(df):
    ''' The function uses nlp_url_features module to return several new url features
        countries_suffix is a df contains all suffixes and their based country name
    '''
    print('\nURL features')
    countries_suffix = pd.read_csv('../../data/classifiers/countries_domain_extention.csv', header=None,
                                   names=['suffix', 'country'])

    # URL features using nlp_url_features module
    # 1. Extract URL
    df['url'] = df.text.apply(lambda x: nlp_url_features.url_extraction(x))
    print('  - url extraction completed')

    # 2. Extract domain extension
    suffixes = countries_suffix.suffix.tolist()
    df['url_country'] = df.text.apply(
        lambda x: nlp_url_features.country_domain_extraction(x, countries_suffix, suffixes))
    print('  - url_country completed')

    # 3. Extract domain
    df['url_domain_name'] = df.url.apply(lambda x: nlp_url_features.domain_extraction(x))
    print('  - url_domain_name completed')

    return df


def num_terms_in_text(tokenized_text, terms_set):
    '''
    :param tokenized_text: list of tokenized words of a text
    :param terms_set: set of tokens
    :return: Number of tokens from terms_set that are in the tokenized_text (without duplications - each token counts as 1)
    '''
    return sum([tokenized_text.count(token) for token in terms_set])


def get_country_support(df):
    # Function gets a dataframe of tweets and returns few features regarding the countries mentioned in the tweet
    # Countries support is a prior for anti/pro-Israel sentiment in a tweet
    # The support is determined by global anti-semitic survey from http://global100.adl.org/about by Global 100 ADL organization
    # The Index Score represents the percentage of adults in this country who answered “probably true" to a majority of the anti-Semitic stereotypes tested'''
    print('  - START: Country support')

    countries_support = country_support.get_countries_support_dict()
    for func in [np.min, np.max, np.average, np.median, np.sum]:
        df['country_support_{}'.format(func.__name__)] = df.tokenized_text.apply(
            lambda x: country_support.get_country_support_score(x, func, countries_support))
    print('    - COMPLETED')
    return df


def get_word_classes_dict():
    # Function returns word tagging dictionary - noun, verb...
    return {
        'CC': 'coordinating conjunction',
        'CD': 'cardinal digit',
        'EX': 'existential there',
        'IN': 'preposition/subordinating conjunction',
        'JJ': 'adjective',
        'JJR': 'adjective, comparative',
        'JJS': 'adjective, superlative',
        'MD': 'modal',
        'NN': 'noun',
        'NNS': 'noun plural',
        'NNP': 'proper noun, singular',
        'NNPS': 'proper noun',
        'POS': 'possessive ending',
        'PRP': 'personal pronoun',
        'PRP$': 'possessive pronoun',
        'RB': 'adverb',
        'RBR': 'adverb, comparative',
        'RBS': 'adverb, superlative',
        'RP': 'particle',
        'TO': 'to',
        'UH': 'interjection',
        'VB': 'verb',
        'VBD': 'verb',
        'VBG': 'verb, gerund/present participle',
        'VBN': 'verb, past participle',
        'VBP': 'verb, sing. present, non-3d',
        'VBZ': 'verb, 3rd person sing. present',
        'WDT': 'wh-determiner',
        'WP': 'wh-pronoun',
        'WRB': 'wh-abverb'}


def creating_word_type(df):
    print('- Word classes features')
    class_dict = get_word_classes_dict()  # class dict format - class_code: explanation
    df['words_types'] = df.tokenized_text.apply(lambda x: word_all_classes(x))
    for key in class_dict.keys():
        df[key] = df['words_types'].apply(lambda x: x.get(key, 0))
    print('  - COMPLETED')
    df.drop(['words_types'], axis=1, inplace=True)
    return df


def word_all_classes(tokenized_text):
    # Extract the words classes (verb, noun...)
    tags = nltk.pos_tag(tokenized_text)
    # Aggregate results - count
    counts = collections.Counter(tag for word, tag in tags)
    total = sum(counts.values())
    class_res = dict((word, float(count) / total) for word, count in counts.items())
    return class_res


# Emotion section - need to be adjusted to online learning
# pkl file needed to be updated every month
# We need to merge the pkl file with the full dataframe, unmatched tweets will have NA instead

def emotion_score(emotion, doc):
    try:
        if type(doc) != dict:
            doc = ast.literal_eval(doc)
        for k in range(3):
            info = doc['document_tone']['tone_categories'][k]['tones']
            for i in info:
                if i['tone_name'] == emotion:
                    return i['score']
    except:
        return np.nan


def emotion_extraction(df, dominant_words=None):
    '''
    :return: a df with all tweets with the emotion feature
    Emotion feature calculated by IBM's Watson API.
    There is a limit of 2,500 text in one month, therefore we save all results in a the csv file, and not calling the API in this function.
    '''
    emotions = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Analytical", "Confident", "Tentative", "Openness",
                "Conscientiousness", "Extraversion", "Agreeableness", "Emotional Range"]

    if 'emotion' not in df.columns and 'Anger' not in df.columns:
        print("  (No 'emotion' column and no emotions features in the df -> merging emotion_df")
        # We need to merge the df with emotion_df since we do not have emotions or the json
        emotion_df = pd.read_pickle('../../data/classifiers/emotion_df.pkl')
        df = df.merge(emotion_df, on='text', how='left', suffixes=('', '_'))[list(df.columns) + emotions]

        # Check if all Tweets have emotion
        df_no_emotion_match = df[df.Anger.isna()]
        print('   Number of Tweets without emotion match:',df_no_emotion_match.shape[0])
        if df_no_emotion_match.shape[0] > 0:
            print('###############################################')
            print('###############################################')
            print('###############################################')
            print('PAY ATTENTION - SOME TWEETS DO NOT HAVE EMOTION')
            print('## ALL TWEETS WITHOUT EMOTIONS WERE REMOVED! ##')
            print('###############################################')
            print('###############################################')
            print('###############################################')
            df = df[df.Anger.notna()]

        # Check now if after the merge we have emotions or just the json
        if 'Anger' not in df.columns:
            print("  (There is only the 'emotion' column in the df -> extracting emotions from json")
            for key in emotions:
                # Extract the emotion from the Watson's doc
                df[key] = df['emotion'].apply(lambda x: emotion_score(key, x))
                # if config.get('emotion multiplication', False):
                #     # Multiplications - we removed by default since it had no value
                #     if key in emotions_multi:
                #         for word_col in words_cols:
                #             df['{}*{}'.format(key, word_col)] = df[key] * df[word_col]
            df = df.drop(columns=['emotion'])

        # Else - nothing to do

    elif 'Anger' in df.columns:
        if 'emotion' in df.column:
            df = df.drop(columns=['emotion'])
        # If we already have all emotions -> return
        print('  (Emotion features already exist - DONE)')
        return df
    else:
        # If we have the json -> we need to convert it to seperate columns:

        # Subset of emotion to be multiplied by all dominant words count
        # emotions_multi = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Openness", "Confident"]
        # words_cols = [col for col in df.columns if col.startswith('word_') and col.endswith('_count')]
        # Create features
        for key in emotions:
            # Extract the emotion from the Watson's doc
            df[key] = df['emotion'].apply(lambda x: emotion_score(key, x))
            # if config.get('emotion multiplication', False):
            #     # Multiplications - we removed by default since it had no value
            #     if key in emotions_multi:
            #         for word_col in words_cols:
            #             df['{}*{}'.format(key, word_col)] = df[key] * df[word_col]

        # Check if all Tweets have emotion
        df_no_emotion_match = df[df.emotion.isna()]
        print('Number of Tweets without emotion match:', df_no_emotion_match.shape[0])
        if df_no_emotion_match.shape[0] > 0:
            print('###############################################')
            print('###############################################')
            print('###############################################')
            print('PAY ATTENTION - SOME TWEETS DO NOT HAVE EMOTION')
            print('## ALL TWEETS WITHOUT EMOTIONS WERE REMOVED! ##')
            print('###############################################')
            print('###############################################')
            print('###############################################')
            df = df[df.Anger.notna()]

        df = df.drop(columns=['emotion'])

    # Adding few features
    df['sum_neg_emotions'] = df[["Anger", "Disgust", "Fear", "Joy", "Sadness"]].sum(axis=1)
    df['sum_neg_emotions * Confident'] = df['sum_neg_emotions']*df['Confident']
    df['sum_neg_emotions * Analytical'] = df['sum_neg_emotions'] * df['Analytical']

    print(' - Emotion feature completed')
    return df