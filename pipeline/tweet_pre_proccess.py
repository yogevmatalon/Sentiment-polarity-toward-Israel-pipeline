
# coding: utf-8

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob

#import the slang module
import slang
import country_support
import nlp_features
    
def strip_punctuation(word):
    punctuation = "!\"$%&\\()*+,./:;<=>?[\\\\]^_`{|}~'#-"
    return ''.join(char for char in word if char not in punctuation)

def spell_correction(corrected_tokens, tokens):
    # Count number of differences in words
    counter = 0 
    for i in range(len(tokens)):
        if corrected_tokens[i]!=tokens[i]:
            counter+=1
    return counter

def domain_stemmer():
    # Use to stem words that Porter stemmer stems badly
    return {'israelis': 'israeli', 'palestinians': 'palestinian', 'palestine':'palestine', 'palestinian':'palestinian', 'zionists':'zionist'}

def get_saved_words(removeSlang = True):
    # Create a set of stop words, countries names, negation words and slang words (if needed) -> saved_words
    # These words won't be stemmed, due to uses of these words in later phases
    saved_words = country_support.get_countries()
    saved_words = saved_words.union(nlp_features.get_negation_words())

    if removeSlang:
        slang_dict = slang.slang_words()
        slang_words = list(slang_dict.keys())
        saved_words = saved_words.union(slang_words)

    return saved_words

def proccess_tweet(df, domain_terms, col='text', removeSlang = True, spellCorrection = False):
    '''
        Function recieves
            - a dataframe with 'text' column to be tokenize
            - a set of domain saved terms to be skipped in the stop-removal (and slang) and stemming phase
        The function apply stemming, stop-words removal, slang correction and tokenization
        Spell correction is currently commented due to low performance 
    '''
    # Initialize a ProterStemmer object
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Create a set of stop words, countries names, negation words and slang words (if needed) -> saved_words
    # These words won't be stemmed, due to uses of these words in later phases
    saved_words = get_saved_words(removeSlang)

    # Tokenization, stemming and stopword (and punctuation) removal - the result is a list of the terms of the tweet.
    # The '#' removed in the strip_punctuation function. Converted to normal word.

    # Convert text to lower case
    df['tokenized_text'] = df[col].apply(lambda text: text.lower())
    print('  - Lower case completed')
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: x.replace('…', ''))
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: x.replace('"', ''))
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: x.replace('\n', ''))

    # Drop links and mentions
    # word_tokenizer(tweet) was replaced by .split(). Reason: probably done some extra pre-proccessing that cause damage
    df['tokenized_text'] = df['tokenized_text'].apply(
        lambda tweet: [strip_punctuation(token) for token in tweet.split() if (
                    (token not in stop_words) and ('http' not in token) and ('pic.twitter' not in token) and ('bitly.' not in token) and (token != 'rt') and (token!= '…')
                    and (token!= '"') and ('bit.ly' not in token) and (not token.startswith('@')))])

    print('  - Strip punctuation completed')
    print('  - Stop-words removal completed')
    print('  - Tokenization completed')

    if removeSlang:
        # Replace slang words
        slang_dict = slang.slang_words()
        df['tokenized_text'] = df['tokenized_text'].apply(lambda tweet: [slang_dict[token] if token in slang_dict else token for token in tweet])
        print('  - Slang words correction completed')

    if spellCorrection:
        # Spell correction for tokens, unless they are domain terms
        df['tokenized_text'] = df['tokenized_text_no_spell'].apply(
            lambda tokens: [str(TextBlob(token).correct()) if token not in domain_terms else token for token in tokens])
        # spelling correction feature
        df['num_spell_errors'] = df[['tokenized_text_no_spell', 'tokenized_text']].apply(lambda x: spell_correction(x[0], x[1]))
        print('  - Spell correction completed')

    # Before using Porter stemmer - use domain stemmer and stem words that Porter stems badly.
    domainStemmer = domain_stemmer()
    # Stem it to words that are in domain_terms, so Porter stemmer will skip these words
    df['tokenized_text'] = df['tokenized_text'].apply(
        lambda tokens: [domainStemmer[token] if token in domainStemmer else token for token in tokens])

    # Stem tokens, unless they are domain terms (or length = 1) or saved words
    df['tokenized_text'] = df['tokenized_text'].apply(
        lambda tokens: [stemmer.stem(token) if (token not in domain_terms and token not in saved_words and len(token)>1) else token for token in tokens])
    print('  - Stemming phase completed')

    # Remove empty tokens
    df['tokenized_text'] = df['tokenized_text'].apply(lambda tokens: [token for token in tokens if len(token) > 1])
    print('  - Cleaning empty tokens completed')

    print('Final number of features: {}'.format(df.shape[0]))
    print('\nCOMPLETED: Pre-proccess')
    print('----------------------\n----------------------')

    return df