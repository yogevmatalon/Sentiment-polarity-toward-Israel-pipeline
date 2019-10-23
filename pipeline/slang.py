import tweet_pre_proccess

def slang_words():
	return {'pee': 'pee', '4': 'for', 'omg': 'oh my god', 'u': 'you', 'tbh' : 'to be honest',
             'imo':'in my opinion', 'icymi': 'in case you miss it', 'snh': 'shake my head','r':'are',
              '2f4u':'too fast for you',
                '4yeo': 'for your eyes only',
                'fyeo': 'for your eyes only',
                'aamof': 'as a matter of fact',
                'ack':'acknowledgment',
                'afaik':'as far as i know',
                'afair':'as far as i remember',
                'afk':'away from keyboard',
                'aka':'also known as',
                'btt':'back to topic',
                'btw':'by the way',
                'bc':'because',
                'cp':'copy and paste',
                'cu':'see you',
                'eod':'end of discussion',
                'fka':'formerly known as',
                'fwiw':'for what its worth',
                'fyo': 'for your information',
                'jfyi': 'just for your information',
                'ftw':'fuck the world',
                'hf':'have fun',
                'hth':'hope this helps',
                'idk':'i dont know',
                'iow':'in other words',
                'lol':'laughing out loud',
                'dgmw': 'dont get me wrong',
                'nntr': 'no need to reply',
                'THX' : 'thanks',
                'TNX':'thanks',
                'fkn': 'fucking',
                'cud': 'could',
                'ICYMI': 'in case you missed it',
                '2moro':'tomorrow',
                '2nte': 'tonight',
                'y': 'why',
                'luv': 'love'}
   
def slang_counter(text):
	# Function gets a text and returns the number of slang words in it
	slang_dict = slang_words()
	text = text.lower()
	text = tweet_pre_proccess.strip_punctuation(text)
	counter = 0
	for word in text.split():
		if word in slang_dict:
			counter+=1
	return counter