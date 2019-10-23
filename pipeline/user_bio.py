from datetime import datetime
import pandas as pd
import tweet_pre_proccess
import pipeline
import domain_entities as de


def get_bio_keywords(domain_terms):
    '''
        Function return a list of important keywords for bio analysis
        These words will be pre-proccessed as 'tweets' in order to avoid mismatch because of stemming.
    '''
    # The original bio keyowrds list
    df_keywords = pd.DataFrame({'text':['israel','palestine', 'activist', 'zionist', 'zion', 'jew', 'jewish', 'justice', 'democracy', 'army', 'jihad', 'student', 'updates', 'news',
                                        'advocate',
                               'racists', 'fascists', 'politics', 'political', 'propaganda', 'revolution', 'truth', 'occupy', 'occupation', 'muslim', 'gaza', 'Middle East', 'injustice',
                               'dictatorship', 'support', 'journalist', 'love', 'UN', 'crime', 'war', 'peace', 'free', 'liberty', 'democrat', 'republican', 'liberator', 'liberate',
                               'politician', 'civil', 'rights', 'fight', 'apartheid', 'terrorist', 'terror', 'freedom', 'resistance', 'book', 'music', 'life', 'hasbara', 'arab']})
    # Apply pre-process
    df_keywords = tweet_pre_proccess.proccess_tweet(df_keywords, domain_terms)
    print('\nbio keywords pre-process')
    # Extract the processed keywords (returned as tokenized text list)
    bio_keywords = list(df_keywords.tokenized_text.apply(lambda x: x[0]))
    return bio_keywords

def user_bio_analysis(bio_df):
    ''' Functions gets a users metadata df and add bio features '''
    print('START: bio features')
    # Get the bio keywords
    domain_terms = de.get_all_domain_terms()
    bio_keywords = get_bio_keywords(domain_terms)
    bio_df.description=bio_df.description.astype('str')

    # Apply pre-process to the df
    print('\nbio_df description column pre-process')
    bio_df = tweet_pre_proccess.proccess_tweet(bio_df, domain_terms, 'description')

    # Adding dummy features for all keywords
    for keyword in bio_keywords:
        # tokenized_text is the tokenized description feature
        bio_df['bio_{}'.format(keyword)] = bio_df.tokenized_text.apply(lambda bio: 1 if keyword in bio else 0)
    # Grouped features
    bio_df['bio_keywords_pro'] = bio_df[['bio_israel','bio_jew','bio_jewish','bio_zion', 'bio_hasbara']].sum(axis=1)
    bio_df['bio_keywords_anti'] = bio_df[['bio_zionist', 'bio_jihad', 'bio_gaza', 'bio_muslim', 'bio_occupi', 'bio_occup', 'bio_apartheid']].sum(axis=1)
    bio_df['bio_keywords_activist'] = bio_df[['bio_activist', 'bio_justice', 'bio_democraci','bio_armi', 'bio_student', 'bio_racist','bio_peace',
                                              'bio_fascist', 'bio_polit', 'bio_propaganda', 'bio_revolut', 'bio_truth', 'bio_middl', 'bio_war',
                                              'bio_injustic', 'bio_dictatorship', 'bio_support','bio_love', 'bio_un', 'bio_civil',
                                              'bio_right', 'bio_fight', 'bio_terror', 'bio_terror', 'bio_freedom', 'bio_resist', 'bio_advoc',
                                              'bio_free', 'bio_liberti', 'bio_democrat', 'bio_republican', 'bio_politician']].sum(axis=1)
    bio_df['bio_keywords_news'] = bio_df[['bio_updat','bio_news','bio_polit','bio_journalist', 'bio_crime', 'bio_book']].sum(axis=1)

    print(' - User bio features COMPLETED')
    return bio_df