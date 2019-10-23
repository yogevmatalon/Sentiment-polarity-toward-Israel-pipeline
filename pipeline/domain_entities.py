
# coding: utf-8

# ## Dicts and lists
# This script holds dictionaries and lists for the classification models

# sources: 
# - https://en.wikipedia.org/wiki/List_of_designated_terrorist_groups
# - https://he.wikipedia.org/wiki/ארגון_טרור

import pandas as pd
import tweet_pre_proccess
path = '../../data/classifiers/'

def entitiy_in_tweet(text, entity_set):
    ''' Function gets a text and set of entities. Returns 1 if the text has one entity (at least) from the set, else 0 '''
    res = sum([1 if token.lower() in text.lower() else 0 for token in entity_set])
    return 1 if res > 0 else 0

def get_pro_orgs():
    # load the csv and returns one column of it as a set
    return set(pd.read_csv(path+'entities_pos_orgs.csv').org_name)

def get_anti_orgs():
    # load the csv and returns one column of it as a set
    # Various sources, mainly manual mining of connections in accounts (e.g. BDS) in Twitter
    # And also: https://www.jpost.com/Diaspora/Antisemitism/Minister-Erdan-BDS-groups-have-ties-to-PFLP-Hamas-and-PA-560368
    return set(pd.read_csv(path+'entities_neg_orgs.csv').org_name)

def get_country_support():
    # load the csv and returns the df
    return pd.read_csv(path+'countries.csv')

def get_all_domain_terms():
    ''' Used for several functions (like pre-proccess) in order to give special treatment and actions for these words (e.g. avoid stemming) '''
    # Create automatically using keywords_dict, politicians (functions below), terrorists and organizations
    terms = set({})
    funcs = [israeli_politics, us_diplomats, other_diplomats, get_domain_keywords, terror_leaders, terror_orgs, specific_terms, hostile_countries]
    for func in funcs:
        terms = terms.union(func())

    # For every term in terms set -> break it to sub-terms:
    # Subterms are seperate terms by ' ', '-'
    # Remove punctuation
    # Convert terms to lowercase
    final_terms = set({})

    for term in terms:
        # term is a string

        # Break the term to all possible subterms (break by '-' or ' ')
        subterms = set(term.split())
        new_subterms = set({})
        for subterm in subterms:
            new_subterms = new_subterms.union(subterm.split('-'))

        subterms = subterms.union(new_subterms)
        for subterm in subterms:
            final_terms.add(tweet_pre_proccess.strip_punctuation(subterm.lower()))
    return final_terms

    # Manual version
    # return {
    #         # politicians
    #         'benjamin', 'netanyahu', 'naftali', 'bennett', 'lieberman', 'obama', 'barak', 'trump', 'bush',
    #         'angela', 'merkel', 'nicolas', 'sarkozy', 'emmanuel','macron', 'françois', 'hollande',
    #         'theresa', 'may', 'david', 'cameron', 'vladimir', 'putin', 'hugo', 'chávez', 'stefan', 'löfven', 'margot','wallström'
    #
    #         #keywords
    #         'israel','gaza','intifada','palestine','palestinian','bds','sjp',
    #         'bdsmovement','worldwithoutwalls','stopthewall','boycottisrael','nationalsjp','prayforisrael','supportisrael','israelvictory',
    #         'boycottisraelnetwork','justice','israeliapartheidweek','apartheidweek','jerusalem','holyland','love_israel',
    #         'loveisrael','hate_israel','freepalestine','supportidf','idf','proisrael','apartheid',
    #         'zionist', #(also 'zionists' for PorterStemmer)
    #         'zion''hamas','hizballah','westernwall', 'nakba',
    #         # Terrorists
    #         'hassan', 'nasrallah', 'tufayli', 'abbas', 'al-musawi', 'musawi',
    #         'ayman', 'alzawahiri', 'zawahiri', 'nasser', 'alwuhayshi', 'qasm', 'alrimi', 'rimi', 'moktar', 'belmoktar',
    #         'abu', 'bakr', 'albaghdadi', 'baghdadi', 'mullah', 'fazlullah',
    #         'abubakar', 'shekau', 'ali', 'khamenei', 'eshaq', 'jahangiri', 'amir', 'hatami', 'vaezi', 'mohammad', 'zarif', 'alavi',
    #         'ali', 'akbar', 'salehi', 'ahmadinejad', 'rouhani', 'bashar','aljolani', 'jolani'
    #         'khaled', 'mashal', 'ahmed', 'yassin', 'aziz', 'alrantisi', 'rantisi', 'ismail', 'haniya', 'mahmoud', 'alzahhar', 'zahhar',
    #
    #         # Terror organizations
    #         'isis', 'hizbollah', 'hezballah', 'hizballah', 'hezbollah', 'hamas', 'hammas',
    #         'islamic', 'jihad', 'muslim', 'bds', 'alqaeda', 'qaeda', 'addin', 'alqassam', 'brigades', 'qassam', 'fatah',
    #         'alaqsa','aqsa', 'tanzim', 'caucasus', 'emirate', 'Ansar', 'bait', 'almaqdis', 'maqdis', 'boko', 'haram', 'taliban', 'ihh',
    #         'kahane', 'chai', 'kach', 'almuslima', 'muslima', 'jabhat', 'alnusra', 'al', 'nusra', 'fida', 'afl', 'pflp', 'delp', 'ppp', 'ppsf', 'plf'
    #         }

def israeli_politics():
    return {'Benjamin Netanyahu', 'Netanyahu', 'Naftali Bennett', 'Bennett', 'Avigdor Lieberman', 'Ehud Barak',
            'Rivlin', 'Reuven Rivlin', 'Shimon Peres', 'Peres'}

def specific_terms():
	return {'peace','campus'}

def us_diplomats():
    return {'Obama', 'Trump', 'Bush','Nicky Hailey', 'Hailey'}

def other_diplomats():
    return {'Angela Merkel', 'Merkel', 
                   'Nicolas Sarkozy', 'Sarkozy', 'Emmanuel Macron', 'Macron', 'François Hollande', 'Hollande',
                   'Pietro Grasso', 'Sergio Mattarella', #italy
                   'Theresa May', 'David Cameron',
                   'Vladimir Putin', 'Nicolás Maduro', 'Hugo Chávez', 'Stefan Löfven', 'Margot Wallström' # venezuela, sweden
           }

def get_domain_keywords():
    return {
            'Israel',
            'Israeli',
            'gaza',
            'intifada',
            'Palestine',
            'Palestinian',
            'BDS',
            'SJP',
            'BDSmovement',
            'WorldWithoutWalls',
            'StopTheWall',
            'BoycottIsrael',
            'NationalSJP',
            'National SJP',
            'Prayforisrael',
            'supportIsrael',
            'startup nation',
            'IsraelVictory',
            'BoycottIsraelNetwork',
            'Justice in Palestine',
            'IsraeliApartheidWeek',
            'Jerusalem',
            'holyland',
            'Love_Israel',
            'LoveIsrael',
            'hate_israel',
            'FreePalestine',
            'supportidf',
            'IDF',
            'JewishLivesMatter',
            'Proisrael',
            'Apartheid',
            'Zionist',
            'zionism',
            'Zion'
            'Hamas',
            'Hizballah',
            'Netanyahu',
            'western wall',
            'nakba',
            'AntiSemitic',
            'AntiSemitism'
            }

def terror_orgs():
    return {
        'ISIS','Hizbollah','Hezballah','Hizballah','Hezbollah','Hamas','Hammas', 'Islamic Jihad', 'Muslim Brotherhood'
        'BDS','al-Qaeda','al Qaeda','Izz ad-Din al-Qassam Brigades','ad-Din al-Qassam','al-Qassam','al Qassam', 'Fatah',
        'Islamic Jihad', 'al-Aqsa Foundation', 'al Aqsa Foundation', 'Islamic State', 'Tanzim', 'Palestinian Islamic Jihad',
        'Caucasus Emirate', 'Ansar Bait al-Maqdis', 'Ansar Bait al Maqdis', 'Force17', 'Boko Haram', 'Taliban', 'IHH',
        'Kach and Kahane Chai', 'Kahane Chai', 'Kach', 'Palestine al-muslima', 'Palestine al muslima', 'Palestine Liberation Front',
        'Jabhat al-Nusra', 'Jabhat al Nusra'
        'FIDA', 
        'ALF',
        'PFLP',
        'DFLP',
        'PPP',
        'PPSF',
        'PLF'	}

# Sources:
# - https://edition.cnn.com/2013/10/28/middleeast/gallery/deadliest-groups-leaders/index.html
# - https://edition.cnn.com/2013/10/09/world/meast/top-10-terrorists/index.html


def terror_leaders():
    return {
            'Hassan Nasrallah', 'Subhi al-Tufayli', 'Abbas al-Musawi', 'Subhi al Tufayli', 'Abbas al Musawi', #Hezbollah
            'Ayman Al-Zawahiri', 'Ayman Al Zawahiri', 'Nasser al-Wuhayshi','Nasser al Wuhayshi', 'Qasm al-Rimi', 'Qasm al Rimi', 'Moktar Belmoktar', #al-Qaeda
            'Abu Bakr al-Baghdadi', 'Abu Bakr al Baghdadi', #ISIS
            'Mullah Fazlullah', # Taliban
            'Abubakar Shekau', # Boko Haram
            'Ali Khamenei', 'Eshaq Jahangiri', 'Amir Hatami', 'Mahmoud Vaezi', 'Mohammad Javad Zarif', 'Mahmoud Alavi', #Iran
            'Ali Akbar Salehi', 'Mahmoud Ahmadinejad', 'Ahmadinejad', 'Hassan Rouhani', 
                # Iran source: https://en.wikipedia.org/wiki/Cabinet_of_Iran
            'Bashar al-Assad', 'Bashar al Assad' #Syria
            'Khaled Mashal', 'Ahmed Yassin', 'Abdel Aziz al-Rantisi', 'Ismail Haniya', 'Mahmoud Al-Zahhar', 'Mahmoud Al Zahhar' #Hamas
            'Abu Mohammed al-Jolani', 'al-Jolani' # al Nusra
        }

def hostile_countries():
    # Source: https://en.wikipedia.org/wiki/Foreign_relations_of_Israel#No_recognition_or_diplomatic_relations
    return {'Iran', 'Turkey', 'Syria', 'Iraq', 'Lebanon', 'North Koera', 'Pakistan', 'Indonesia', 'Kuwait',
            'Afghanistan', 'Oman', 'Qatar', 'Yemen', 'Venezuela', 'Sudan', 'Somalia', 'Libya', 'Malaysia', 'Algeria'}

def news_companies():
    return {'ABC', 'NBC', 'FOX', 'CNN', 'CBS', 'BBC', 'Yediot', 'Ynet', 'Reuters', 'Bloomberg', 'al-jazeera', 'jazeera',
            'The Washington Post', 'chicago tribune', 'dailymail', 'telegraph', 'forbes', 'guardian', 'huffington post',
            'irishtimes','irish times', 'israelhayom', 'israel hayom', 'Jerusalem Post', 'jpost', 'Haaretz', 'nytimes',
            'newyorker', 'timesofindia', 'timesofisrael', 'washingtonpost', 'ynetnews','friendfeed'}

# Anti orgs (not terror org) - csv
# Sources: 
# - https://www.adl.org/sites/default/files/documents/assets/pdf/israel-international/Top-10-Anti-Israel-Groups-in-America.pdf