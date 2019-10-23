# coding: utf-8
import numpy as np
import http.client
import urllib.parse
import requests
import re

def url_extraction(text, expand_full=False):
    ''' Returns the url in the text, if exists '''
    try:
        text = str(text)
        if 'http://' not in text and 'https://' not in text and 'www.' not in text:
            return np.nan
        elif 'http://' not in text and 'https://' not in text and 'www.' in text:
            # Just a 'www...' link without http/s
            url = text[text.find('www.'):].split(' ')[0]
            return strip_url_punctuation(url if '...' not in url else url.split('...')[0])
        elif 'bit.ly' in text or 'bitly.' in text or 'goo.' in text or 'tinyurl' in text \
                or 'ow.ly' in text or '/fb.' in text or '/tl.gd' in text or 'jijr.com' in text:
            try:
                # Url requires expansion
                if 'http://' in text or 'https://' in text:
                    url = text[text.find('http://'):].split(' ')[0] if 'http://' in text else text[text.find('https://'):].split(' ')[0]
                elif 'www.' in text:
                    url = text[text.find('www.'):].split(' ')[0]
                else:
                    return np.nan
                url = url.split('...')[0] if '...' in url else url
                return expand_url(strip_url_punctuation(url)) if not expand_full else expand_url_full(strip_url_punctuation(url))
            except:
                if 'http://' in text or 'https://' in text:
                    url = text[text.find('http://'):].split(' ')[0] if 'http://' in text else text[text.find('https://'):].split(' ')[0]
                    # Return the http link only if it's longer than 'https'
                    url = url.split('...')[0] if '...' in url else url
                    return strip_url_punctuation(url) if len(url) > 5 else np.nan
                else:
                    url = text[text.find('www.'):].split(' ')[0]
                    url = url.split('...')[0] if '...' in url else url
                    return strip_url_punctuation(url)
        elif ('http://' in text or 'https://' in text):
            url = text[text.find('http//'):].split(' ')[0] if 'http://' in text else text[text.find('https://'):].split(' ')[0]
            # Return the http link only if it's longer than 'https'
            url = url.split('...')[0] if '...' in url else url
            return strip_url_punctuation(url) if len(url) > 5 else np.nan
    except IndexError as e:
        return np.nan

def expand_url_full(url):
    '''
        Function get a shortened url (string)  and return the source url (string):
        e.g. http://bit.ly/rgCbf  => http://webdesignledger.com/freebies/the-best-social-media-icons-all-in-one-place

        This version is more accurate, but takes much more time. Not scalable.
    '''
    session = requests.Session()  # so connections are recycled
    resp = session.head(url, allow_redirects=True)
    return resp.url

def expand_url(url):
    '''
        Function get a shortened url (string)  and return the source url (string):
        e.g. http://bit.ly/rgCbf  => http://webdesignledger.com/freebies/the-best-social-media-icons-all-in-one-place
    '''
    try:
        parsed = urllib.parse.urlparse(url)
        h = http.client.HTTPConnection(parsed.netloc)
        h.request('HEAD', parsed.path)
        response = h.getresponse()
        if response.status//100 == 3 and response.getheader('Location'):
            return response.getheader('Location')
        else:
            return url
    except UnicodeEncodeError as e:
        # There are some foreign Unicode characters.
        # Parse as ascii, the 'ignore' part will tell it to just skip those characters.
        url_fixed = url.encode('ascii', 'ignore')
        parsed = urllib.parse.urlparse(url)
        h = http.client.HTTPConnection(parsed.netloc)
        h.request('HEAD', parsed.path)
        response = h.getresponse()
        if response.status//100 == 3 and response.getheader('Location'):
            return response.getheader('Location')
        else:
            return url
    except:
        return url

def strip_url_punctuation(text):
    '''
        Function gets a url text and return it without punctuation
    '''
    punctuation = "!\"$%&\\()*+,;^`{|}~'"
    return ''.join(char for char in text if char not in punctuation)

# ####################################################################
# OLD method - was deplaced due to poor performance (run time)

# def expand_url(shortened_url):
#     '''
#         Function get a shortened url (string)  and return the source url (string):
#         e.g. http://bit.ly/rgCbf  => http://webdesignledger.com/freebies/the-best-social-media-icons-all-in-one-place
#     '''
#     try:
#         conn = pycurl.Curl()
#         conn.setopt(pycurl.URL, shortened_url)
#         conn.setopt(pycurl.FOLLOWLOCATION, 1)
#         conn.setopt(pycurl.CUSTOMREQUEST, 'HEAD')
#         conn.setopt(pycurl.NOBODY, True)
#         conn.perform()
#         return conn.getinfo(pycurl.EFFECTIVE_URL)
#     except UnicodeEncodeError as e:
#         # There are some foreign Unicode characters.
#         # Parse as ascii, the 'ignore' part will tell it to just skip those characters.
#         shortened_url_fixed = shortened_url.encode('ascii', 'ignore')
#         conn = pycurl.Curl()
#         conn.setopt(pycurl.URL, shortened_url_fixed)
#         conn.setopt(pycurl.FOLLOWLOCATION, 1)
#         conn.setopt(pycurl.CUSTOMREQUEST, 'HEAD')
#         conn.setopt(pycurl.NOBODY, True)
#         conn.perform()
#         return conn.getinfo(pycurl.EFFECTIVE_URL)
#     except:
#         return shortened_url
#
# ####################################################################

def domain_extraction(url):
    """ Gets a url and returns the domain name of the url in the tweet, if exists """
    try:
        if url is not np.nan:
            # Extract domain
            if 'http' in url:
                domain = url.split('//')[1].split('.')
                if domain[0] in ['www', 'm', 'he', 'de', 'en', 'es', 'edition', 'mobile', 'mob','dl-web','on']:
                    return domain[1]
                else:
                    return domain[0]
            else:
                return url.split('.')[1].split('.')[0]
        else:
            return np.nan
    except IndexError as e:
        return np.nan


def country_domain_extraction(text, countries_suffix, suffixes):
    """ Gets a text and returns the country of the url in the tweet, identified by the domain extension """
    text = str(text)
    url = url_extraction(text)
    try:
        if url is not np.nan:
            url = str(url)

            # Patch
            # Handle specific cases that logic misses:
            #   1. 'co.il' -> Israel (misidentified as 'co' -> Columbia)
            #   2. 'com.ar' -> Argentina (misidentified as 'com')
            if '.co.il' in url:
                return 'Israel'
            if '.com.ar' in url:
                return 'Argentina'
            if 'bit.ly' in url:
                return 'bitly'
            if '/goo.' in url or '.goo.' in url:
                return 'US Commercial'
            if 'http' in url:
                url = url.split('//')[1].split('.')
            else:
                # url with 'www' only
                url = url.split('www.')[1].split('.')

            # Split by '/'. Relevant for cases such as http://citifmonline.com/index.php -> com, index -> com
            url = [sub_url.split('/')[0] for sub_url in url]

            # try for sub-domains first
            for suff in suffixes:
                sub_domain = suff + '/'
                if sub_domain in url:
                    return countries_suffix[countries_suffix.suffix == suff].country.tolist()[0]

            # if sub-domain not found -> search without '/'
            for suff in suffixes:
                if suff in url:
                    return countries_suffix[countries_suffix.suffix == suff].country.tolist()[0]
        else:
            return np.nan
    except IndexError as e:
        return np.nan

def find_original_tweet(tweet):
    '''
        Function gets a tweet text and find the original tweet id it RTed (if it has a RT)
    '''
    #print('tweet: ', tweet)

    # Find all url segments in the tweet (https:// or http://)
    url_points = [m.start() for m in re.finditer('https?://', tweet)]
    if len(url_points) > 0:
        # If at least one url has been found, run on all urls
        for i in url_points:
            # Trim the tweet from the left at the beginning of the i url
            cur_tweet_segment = tweet[i:]
            #print('cur_tweet_segment: ', cur_tweet_segment)

            # Extract the url of this segment (also expand it if it shortened)
            url = url_extraction(cur_tweet_segment)
            #print('url: ', url)

            if url is not np.nan:
                # If this a url of a tweet, it should contain 'https://twitter.com/' in it
                original_tweet_url = url if 'https://twitter.com/' in url or 'http://twitter.com/' in url else np.nan
                #print('original_tweet: ', original_tweet_url)

                original_tweet_id = int(original_tweet_url[original_tweet_url.find('/statuses') + 1:].split('/')[1].split('?')[
                                            0]) if original_tweet_url is not np.nan else np.nan
                #print('original_tweet_id: ', original_tweet_id)

                if original_tweet_id is not np.nan:
                    return int(original_tweet_id)

    # No RT found
    #print('No RT in the tweet')
    return np.nan