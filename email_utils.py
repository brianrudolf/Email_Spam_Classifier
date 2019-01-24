import os, re
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# needed on first run:
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

def separate_subjects(messages):
	subject = []
	body = []
	for message in messages:		
		subject.append(' '.join(message[0].split(' ')[1:]))
		body.append(' '.join(message[1:]))
	return subject, body

def clean_text(data):
	tokened = []

	mapping = {ord(c): None for c in '=!?#*'}	# define special characters to remove

	web = re.compile(r'(http|https)://[^\s]*') 	# regular expression for replacing web addresses
	number = re.compile(r'[0-9]+')				# regular expression for replacing numbers
	email_addr = re.compile(r'[\S]+@[\S]+')		# regular expression for replacing email addresses
	dollar = re.compile(r'[$]+')				# regular expression for replacing email addresses
	http = re.compile(r'<[^<>]+>') 				# regular expression for replacing http 

	for content in data:
		# remove special characters
		content = content.translate(mapping)

		# clean text using regex substitutions
		content = web.sub('http', content)
		content = number.sub('number', content)
		content = email_addr.sub('email_addr', content)
		content = dollar.sub('dollar', content)  
		content = http.sub(' ', content)

		content = content.replace('&nbsp', '')	    

		# tokenize each subject (turn string into a list of strings)
		sentence = word_tokenize(content)	    

		# lower case all words, lemmatize each word (similar to stem) to reduce words to their 'root'
		lem_sent = []        
		for word in sentence:
			word = word.lower()
			# only keep words longer than two letters, and ones that aren't in 'stopwords' 
			if len(word) > 2 and word not in stop_words:
				# alternative to stemming
				lem_sent.append(wnl.lemmatize(word))

		lem_sent = ' '.join(lem_sent)
		tokened.append(lem_sent)	

	return tokened