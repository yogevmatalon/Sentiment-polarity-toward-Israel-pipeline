import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm, tqdm_notebook

tqdm.pandas(tqdm_notebook())

def import_network_data(path='../../data/network_data/'):
    print(' - START: Import network data')
    # Init dict of counters for filtered number of users
    filtered_users_counters = {}

    # Load network data
    # with open(path+'df_links.pkl', 'rb') as handle:
    #     df_links = pickle.load(handle)
    df_links = pd.read_pickle(path+'df_links.pkl')

    print('  - df_links imported')
    # with open(path+'df_nodes.pkl', 'rb') as handle:
    #     df_nodes = pickle.load(handle)
    df_nodes = pd.read_pickle(path + 'df_nodes.pkl')

    print('  - df_nodes imported')

    # First step - loading identities lists from csv files
    # Load all users of all types into data frames
    # All files with headers - | screen_name | User Name |
    #  Activists - user who wrote the domain data (800k tweets about Israel domain)
    print(' - Building user_identities dictionary')
    activists = pd.read_csv(path+'activists.csv')
    activists['identity'] = 'activist'

    # Pro and Neg activists. Sources:
        # https://twitter.com/URJorg/lists/urj-rabbis/members - Rabbis list in Twitter
        # Manual search
        # Canary mission website - for Neg activists
    activists_pos = pd.read_csv(path + 'activists_pos.csv')
    activists_pos['screen_name'] = activists_pos['screen_name'].apply(lambda x: x.lower())
    activists_pos['identity'] = 'activist_pos'
    # Neg
    activists_neg = pd.read_csv(path + 'activists_neg.csv')
    activists_neg['screen_name'] = activists_neg['screen_name'].apply(lambda x: x.lower())
    activists_neg['identity'] = 'activist_neg'

    neutral_users = pd.read_csv(path+'neutral_users.csv')
    neutral_users['screen_name'] = neutral_users['screen_name'].apply(lambda x: x.lower())
    neutral_users['identity'] = 'neutral'

    # Organizations:
    #  1. Well known organizations (basic knowledge)
    #  2. Manual search in twitter - followers of well known organizations (such as BDS branches, SJP branches...)
    #  3. Internet:
        # https://en.wikipedia.org/wiki/Israel_lobby_in_the_United_States
        # http://www.jcouncil.org/site/DocServer/Jewish_Campus_Resources_Fall.pdf?docID=9861
        # https://www.independent.co.uk/news/world/middle-east/israel-bds-blacklist-banned-barred-entry-boycott-divestment-sanctions-palestine-solidarity-campaign-a8146686.html
        # https://en.wikipedia.org/wiki/Anti-Israel_lobby_in_the_United_States
        # https://www.timesofisrael.com/the-25-most-influential-people-on-jewish-twitter/
        # http://www.barakabits.com/2014/08/5-must-follow-twitter-accounts-palestine
    org_pos = pd.read_csv(path+'org_pos.csv')
    org_pos['screen_name'] = org_pos['screen_name'].apply(lambda x: x.lower())
    org_pos['identity'] = 'org_pos'

    org_neg = pd.read_csv(path+'org_neg.csv')
    org_neg['screen_name'] = org_neg['screen_name'].apply(lambda x: x.lower())
    org_neg['identity'] = 'org_neg'

    campuses = pd.read_csv(path+'campuses.csv')
    campuses['screen_name'] = campuses['screen_name'].apply(lambda x: x.lower())
    campuses['identity'] = 'campus'

    users = pd.concat((activists, org_pos, org_neg, campuses, activists_pos, activists_neg, neutral_users), axis=0)

    # Define dictionaries that holds screen_names and their type: activist, organization (positive/negative) and campus user
    # key - user screen_name, value - user type
    # Key not in dict -> the user is a student
    user_identities = {}
    for i in tqdm_notebook(range(users.shape[0])):
        user_identities[users.iloc[i, 0]] = users.iloc[i, 2]
        # Increase relevant counter in 1 user
        filtered_users_counters[users.iloc[i, 2]] = filtered_users_counters.get(users.iloc[i, 2], 0)+1

    # Add assumed activists - activist_pos and activist_neg. Those activists are detemined by hashtags.
    df_tweets = pd.read_csv('../../data/domain_full_data.tsv', sep='\t', header=0)
    assumed_pos_activists, assumed_neg_activists = find_activists(df_tweets, num_hashtag_filter=1)

    # Add to user_identities
    for user in assumed_pos_activists:
        user_identities[user] = 'activist_pos'
        filtered_users_counters['activist_pos'] = filtered_users_counters.get('activist_pos', 0)+1
    for user in assumed_neg_activists:
        user_identities[user] = 'activist_neg'
        filtered_users_counters['activist_neg'] = filtered_users_counters.get('activist_neg', 0)+1


    print('Entities in the network:')
    print(' - {} pro-Israel organizations ({} filtered), {} anti-Israel organization ({} filtered)'.format(org_pos.shape[0], filtered_users_counters.get('org_pos',0), org_neg.shape[0], filtered_users_counters.get('org_neg',0)))
    print(' - {} pro-Israel activists ({} filtered), {} anti-Israel activists ({} filtered)'.format(activists_pos.shape[0]+len(assumed_pos_activists), filtered_users_counters.get('activist_pos',0), len(assumed_neg_activists)+activists_neg.shape[0], filtered_users_counters.get('activist_neg')))
    print(' - {} domain activists'.format(activists.shape[0], filtered_users_counters.get('activist',0)))
    print(' - {} US-campuses ({} filtered)'.format(campuses.shape[0], filtered_users_counters.get('campus',0)))
    print(' COMPLETED: Import network data')
    return df_nodes, df_links, user_identities

def get_activists_hashtags():
    '''
    :return: list of tuples (hashtag, 'neg'/'pos') for classifying activists to be pro or anti israel
    '''
    hashtags = [('#supportisrael', 'pos'),
                ('#iloveisrael', 'pos'),
                ('#proisrael', 'pos'),
                ('#loveisrael', 'pos'),
                ('#supportidf', 'pos'),
                ('#blessisrael', 'pos'),
                ('#foreverisrael', 'pos'),
                ('#bdsfail', 'pos'),
                ('#nosuchplaceaspalestine','pos'),
                ('#amisraelchai', 'pos'),
                ('#standwithisrael', 'pos'),
                ('#israelilivesmatter', 'pos'),
                ('#defeathamas', 'pos'),
                ('#israelforever', 'pos'),
                ('#wesupportisrael', 'pos'),
                ('#jewishlivesmatter', 'pos'),
                ('#fuckhamas', 'pos'),
                ('#israel4peace', 'pos'),
                ('#istandwithisrael', 'pos'),
                ('#israelunderattack', 'pos'),
                ('#antiisrael', 'neg'),
                ('#greatreturnmarch', 'neg'),
                ('#israeliapartheidweek', 'neg'),
                ('#apartheidisrael', 'neg'),
                ('#boycotteurovision', 'neg'),
                ('#freegaza', 'neg'),
                ('#prayforgaza', 'neg'),
                ('#boycottisrael', 'neg'),
                ('#gazaunderattack', 'neg'),
                ('#prayforpalestine', 'neg'),
                ('#freepalestine', 'neg')]
    return hashtags

def search_hashtags(pos_activists, neg_activists, tweet, hashtags):
    '''
        Function searches for all hashtags in the tweet's text.
        If found, it adds the hashtags to the user's activists hashtags histogram.
    '''
    # Iterate over all hashtags
    text = tweet.text.lower()
    for hashtag, hashtag_support in hashtags:
        if hashtag.lower() in text:
            # If a hashtag from the list found, add it, and its user to the appropriate dictionary
            if hashtag_support == 'pos':
                # Add the hashtag to the user dictionary: key=screen_name, value={total_hashtag_count, hashtag dict}
                    # hashtag_dict: value = hashtag, value = hashtag_count
                pos_activists[tweet.screen_name] = pos_activists.get(tweet.screen_name, {'count':0, 'hashtags':{}})
                pos_activists[tweet.screen_name]['hashtags'][hashtag] = pos_activists[tweet.screen_name]['hashtags'].get(hashtag, 0)
                pos_activists[tweet.screen_name]['count'] += 1
                pos_activists[tweet.screen_name]['hashtags'][hashtag] += 1
            elif hashtag_support == 'neg':
                neg_activists[tweet.screen_name] = neg_activists.get(tweet.screen_name, {'count':0, 'hashtags':{}})
                neg_activists[tweet.screen_name]['hashtags'][hashtag] = neg_activists[tweet.screen_name]['hashtags'].get(hashtag, 0)
                neg_activists[tweet.screen_name]['count'] += 1
                neg_activists[tweet.screen_name]['hashtags'][hashtag] += 1

def find_activists(df_tweets, num_hashtag_filter=1):
    '''
        Input: a data frame of domain tweets. List of politic hashtags is supplied by get_activists_hashtags function.
            - Each hashtag get a classification of positive or negative (regarding israel).
        The function returns a dictionary of assumed pro/anti activists, by searching for positive/negative hashtags in their tweets.
        Output: dictionary, where key=screen_name, value = list of pro/anti-Israel used hashtags
    '''
    hashtags = get_activists_hashtags()
    pos_activists = {}
    neg_activists = {}
    print('Finding activists using hashtags')
    df_tweets[['screen_name', 'text']].apply(lambda tweet: search_hashtags(pos_activists, neg_activists, tweet, hashtags), axis=1)

    if type(num_hashtag_filter) is not dict: # Meaning num_hashtag_filter is an int which is relevant for both positive and negative activists
        # Filter activists with total hashtag count < num_hashtag_filter:
        pos_activists_filtered = {user for user in pos_activists if pos_activists[user]['count'] >= num_hashtag_filter}
        neg_activists_filtered = {user for user in neg_activists if neg_activists[user]['count'] >= num_hashtag_filter}
    else:
        # num_hashtag_filter can be a dict with different num for positive activists and negative activists
        pos_activists_filtered = {user for user in pos_activists if pos_activists[user]['count'] >= num_hashtag_filter['pos']}
        neg_activists_filtered = {user for user in neg_activists if neg_activists[user]['count'] >= num_hashtag_filter['neg']}

    print('Pro activists found: {}, {} after filtering users with not enough hashtags'.format(len(pos_activists.keys()),len(pos_activists_filtered)))
    print('Anti activists found: {}, {} after filtering users with not enough hashtags'.format(len(neg_activists.keys()),len(neg_activists_filtered)))

    # See if there are screen_names that appear in both lists
    activists_intersection = neg_activists_filtered.intersection(pos_activists_filtered)
    print('Activists groups intersection', activists_intersection)

    # Remove intersection from both dicts, if user in dict
    for user in activists_intersection:
        pos_activists_filtered.remove(user)
        neg_activists_filtered.remove(user)

    return pos_activists_filtered, neg_activists_filtered

def links_majority(pos,neg):
    if pos==neg:
        return 'tie'
    else:
        return 'pos' if pos>neg else 'neg'


def create_network_features(df_nodes, df_links, user_identities, target=''):
    ''''
        Function recieves df_nodes, df_links, and user_identities dictionary.
        The function return a data-frame of features for each user (node), using his links in the network.
    '''
    print(' - START: Creating network features')
    print('   - Features by link')
    df_links['is_pos_link'] = df_links.followed.progress_apply(
        lambda followed_user: 1 if followed_user in user_identities and user_identities[followed_user] in ['org_pos',
                                                                                                           'activist_pos'] else 0)
    df_links['is_neg_link'] = df_links.followed.progress_apply(
        lambda followed_user: 1 if followed_user in user_identities and user_identities[followed_user] in ['org_neg',
                                                                                                           'activist_neg'] else 0)
    df_links['is_pos_org_link'] = df_links.followed.progress_apply(
        lambda followed_user: 1 if followed_user in user_identities and user_identities[followed_user] =='org_pos'
                                                                                                            else 0)
    df_links['is_neg_org_link'] = df_links.followed.progress_apply(
        lambda followed_user: 1 if followed_user in user_identities and user_identities[followed_user] == 'org_neg'
                                                                                                            else 0)
    df_links['is_activist_link'] = df_links.followed.progress_apply(
        lambda followed_user: 1 if followed_user in user_identities and user_identities[
            followed_user] in ['activist','activist_neg','activist_pos'] else 0)

    print('   - Aggregating features by link to user level')
    # Network_data shows for each user how many user he follows who are activists/pro/neg (our degree)
    network_data = df_links.groupby('follower').agg(
        {'is_pos_link': 'sum', 'is_neg_link': 'sum', 'is_neg_org_link': 'sum', 'is_pos_org_link':'sum',
         'is_activist_link': 'sum', 'followed': np.size}).reset_index()
    network_data.rename(columns={'follower':'user',
                                 'is_pos_org_link':'total_pos_orgs_links',
                                 'is_neg_org_link':'total_neg_orgs_links',
                                 'is_pos_link': 'total_pos_links',
                                 'is_neg_link': 'total_neg_links',
                                 'is_activist_link': 'total_activists_links',
                                 'followed': 'total_out_links'
                               }, inplace=True)

    # Add features - ratios
    print('   - Adding ratio features')
    network_data['pos_links_ratio'] = network_data['total_pos_links'] / network_data['total_out_links']
    network_data['neg_links_ratio'] = network_data['total_neg_links'] / network_data['total_out_links']
    network_data['activists_links_ratio'] = network_data['total_activists_links'] / network_data['total_out_links']
    network_data['links_majority'] = network_data.apply(lambda x: links_majority(x.total_pos_links,x.total_neg_links), axis=1)

    # Is the user anti/pro Israel according to our records?
    # if target not in ['user_support','user_support_network']:
    #     network_data['is_anti_user'] = network_data.user.apply(lambda x: 1 if user_identities.get(x,False) in ['org_neg','activist_neg'] else 0)
    #     network_data['is_pro_user'] = network_data.user.apply(lambda x: 1 if user_identities.get(x,False) in ['org_pos','activist_pos'] else 0)

    # Average degree of neighbors
    print('   - Average degree of neighbors features')
    ## In-degree for each user = user impact in the network
    df_in = df_links[['followed','follower']].groupby('followed').agg({'follower':np.size}).reset_index().rename(columns={'followed':'user', 'follower':'in_degree'})
    df_out = df_links[['followed','follower']].groupby('follower').agg({'followed':np.size}).reset_index().rename(columns={'follower':'user', 'followed':'out_degree'})

    df_knn = df_in.merge(df_out, on='user')
    # Fill NA - in case user does not have friends/followers
    df_knn = df_knn.fillna(0)

    df_knn = df_links.merge(df_knn, left_on = 'follower', right_on='user')

    # Export df_in as dict -> will be used to add the in_degree later
    df_in = df_in[['user','in_degree']].set_index(keys='user')
    df_in = df_in.to_dict(orient='index')

    # Average all in-degrees
    df_knn = df_knn[['followed','in_degree','out_degree']].groupby('followed').mean().reset_index()
    df_knn.rename(columns={'in_degree': 'knn_in', 'out_degree':'knn_out'}, inplace=True)

    # Merge results with network_data
    network_data = network_data.merge(df_knn, left_on='user', right_on='followed')
    network_data.rename(columns={'knn_in':'average_in_knn', 'knn_out':'average_out_knn'}, inplace=True)

    # Add in_degree (if no value - user has no followers)
    network_data['in_degree'] = network_data.followed.apply(lambda name: df_in.get(name,{'in_degree': 0})['in_degree'])
    # Rename total_out_links to out_degree
    network_data.rename(columns={'total_out_links': 'out_degree'}, inplace=True)

    network_data.drop(['followed'], inplace=True, axis=1)
    network_data = network_data.set_index(keys='user')

    # Export to pickle
    with open('../../data/classifiers/network_features_data.pickle', 'wb') as handle:
        pickle.dump(network_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('  - COMPLETED: Creating network features')
    return network_data

def add_network_features(df, load_network_data = True, path='../../data/network_data/', target=''):
    ''''
        Function recieves data df
        The function return a data-frame of features for each user (node), using his links in the network.
        The function try to access a pre-made output (in case this long proccess has been done already).
        If the pre-made file is not exist, it creates it (using create_network_features function and import data using import_network_data), and saves it.
    '''
    if load_network_data:
        # Try to find the file
        import_ok = False
        try:
            print('Try to import network features data')
            with open('../../data/classifiers/network_features_data.pickle', 'rb') as handle:
                network_data = pickle.load(handle)
            import_ok = True

        except (FileNotFoundError) as e:
            print('Network features data is not exist, creating it now.')
            df_nodes, df_links, user_identities = import_network_data(path)
            network_data = create_network_features(df_nodes, df_links, user_identities, target)

        network_data_cols = list(network_data.columns)

        print(' - Adding network features')
        df = df.merge(network_data, left_on='screen_name', right_index=True, how='left')

        # Fill NaNs with zero - if we don't know about the connections of this user, we have to assume s/he is neutral.
        df[network_data_cols] = df[network_data_cols].fillna(0)
        df['links_majority'] = df['links_majority'].apply(lambda x: 'Unknown' if x==0 else x)
        dummy = pd.get_dummies(df['links_majority'])
        df = pd.concat([df, dummy], axis=1)
        df.drop(['links_majority'], inplace=True, axis=1)
        df.rename(columns={'pos': 'pos_links_majority', 'neg': 'neg_links_majority'}, inplace=True)

        if not import_ok:
            # Export to pickle
            with open('../../data/classifiers/network_features_data.pickle', 'wb') as handle:
                pickle.dump(network_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('COMPLETED: Network features')
        return df

    else:
        print('load_network_data=False. Creating network features data now.')
        df_nodes, df_links, user_identities = import_network_data(path)
        network_data = create_network_features(df_nodes, df_links, user_identities, target)
        network_data_cols = list(network_data.columns)

        print(' - Adding network features')

        df = df.merge(network_data, left_on='screen_name', right_index=True, how='left')
        # Fill NaNs with zero - if we don't know about the connections of this user, we have to assume s/he is neutral.
        df[network_data_cols] = df[network_data_cols].fillna(0)
        df['links_majority'] = df['links_majority'].apply(lambda x: 'Unknown' if x==0 else x)
        dummy = pd.get_dummies(df['links_majority'])
        df = pd.concat([df, dummy], axis=1)
        df.drop(['links_majority'], inplace=True, axis=1)
        df.rename(columns={'pos': 'pos_links_majority', 'neg': 'neg_links_majority'}, inplace=True)

        # Export to pickle
        with open('../../data/classifiers/network_features_data.pickle', 'wb') as handle:
            pickle.dump(network_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('COMPLETED: Network features')
        return df

