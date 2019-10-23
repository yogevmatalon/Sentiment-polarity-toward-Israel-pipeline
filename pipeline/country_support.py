import numpy as np

def get_countries():
    return set(get_countries_support_dict().keys())

def get_countries_support_dict():
    return {'argentina': 0.24, 'chile': 0.37, 'uruguay': 0.33, 'paraguay': 0.35, 'bolivia': 0.3,
                    'brazil': 0.16, 'peru': 0.38, 'colombia': 0.41, 'venezuela': 0.3, 'panama': 0.52,
                    'costa rica': 0.32, 'nicaragua': 0.34, 'guatemala': 0.36, 'mexico': 0.24, 'jamaica': 0.18,
                    'haiti': 0.26, 'usa': 0.09, 'us': 0.09, 'canada': 0.14, 'iceland': 0.16, 'norway': 0.15,
                    'sweden': 0.04, 'finland': 0.15, 'denmark': 0.09, 'united kingdom': 0.08,
                    'unitedkingdom': 0.08, 'britain': 0.08, 'ireland': 0.2, 'nethrlands': 0.05, 'belgium': 0.27,
                    'germany': 0.27, 'france': 0.37, 'switzerland': 0.26, 'austia': 0.28, 'portugal': 0.21,
                    'spain': 0.29, 'italy': 0.2, 'greece': 0.69, 'estonia': 0.22, 'latvia': 0.28,
                    'lithuania': 0.36, 'belarus': 0.38, 'poland': 0.45, 'czechrepublic': 0.13,
                    'czech republic': 0.13, 'russia': 0.3, 'slovenia': 0.27, 'croatia': 0.33, 'hungary': 0.41,
                    'romania': 0.35, 'moldova': 0.3, 'ulraine': 0.38, 'bulgaria': 0.44, 'sebia': 0.42,
                    'montenegro': 0.29, 'australia': 0.14, 'new zealand': 0.14, 'newzealand': 0.14,
                    'japan': 0.23, 'south korea': 0.53, 'southkorea': 0.53, 'china': 0.2, 'mongolia': 0.26,
                    'kazakhstan': 0.32, 'georgia': 0.32, 'armenia': 0.58, 'azerbaijan': 0.37, 'india': 0.2,
                    'bangladesh': 0.32, 'laos': 0.002, 'thailand': 0.13, 'vietnam': 0.06, 'philippines': 0.03,
                    'malaysia': 0.61, 'singapore': 0.16, 'indonesia': 0.48, 'senegal': 0.53, 'ghana': 0.15,
                    'nigeria': 0.16, 'cameroon': 0.35, 'uganda': 0.16, 'kenya': 0.35, 'tanzania': 0.12,
                    'botswana': 0.33, 'south africa': 0.38, 'southafrica': 0.38, 'mauritius': 0.44, 'turkey': 0.69,
                    'marocco': 0.8, 'algeria': 0.87, 'tunisia': 0.86, 'libya': 0.87, 'egypt': 0.75, 'yemen': 0.88,
                    'saudi arabia': 0.74, 'saudiarabia': 0.74, 'saudi': 0.74, 'saudia': 0.74, 'oman': 0.76,
                    'qatar': 0.8, 'bahrain': 0.81, 'kuwait': 0.82, 'iraq': 0.92, 'iran': 0.56, 'jordan': 0.81,
                    'gaza': 0.93, 'westbank': 0.93, 'west bank': 0.93, 'lebanon': 0.78}

def get_country_support_score(tokenized_text, func, countries_support):
    # Functions gets tokenized_text and find all countries in it. Return function of all their support (avg, min...).
    # Search for all countries in
    countries_in_text = [countries_support[country] for country in countries_support.keys() if country in tokenized_text]
    res = func(countries_in_text) if len(countries_in_text) > 0 else 0
    return res