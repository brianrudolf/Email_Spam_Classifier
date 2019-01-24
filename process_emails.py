import os, pickle

from email_utils import clean_text, separate_subjects

# Variables to gather ham and spam emails
ham_messages = []
spam_messages = []

# loop through the 6 folders of ham/spam emails
for i in range(1,7):

	os.chdir('/home/brian/ML_Projects/Email_Datasets/Enron/preprocessed/enron{}/ham'.format(i))
	ham_emails = os.listdir()

	for email in ham_emails:
		# ignore characters that can't be read (emoticons)
	    with open(email, 'r', errors='ignore') as f:        
	        ham_messages.append(list(f))
	print(len(ham_messages), " ham emails")

	os.chdir('/home/brian/ML_Projects/Email_Datasets/Enron/preprocessed/enron{}/spam'.format(i))
	spam_emails = os.listdir()

	for email in spam_emails:
		# ignore characters that can't be read (emoticons)
	    with open(email, 'r', errors='ignore') as f:
	        spam_messages.append(list(f))
	print(len(spam_messages), " spam emails")

#	Process message subject and body text
#	-> clean up emails into more readable data for algorithms

ham_subjects, ham_bodies = separate_subjects(ham_messages)
spam_subjects, spam_bodies = separate_subjects(spam_messages)

# Change directory to save processed emails
os.chdir('/home/brian/ML_Projects/Email_Datasets/email_classifier')

clean_ham_subjects 	= clean_text(ham_subjects)
pickle_out = open("clean_ham_subjects.pickle", 'wb')
pickle.dump(clean_ham_subjects, pickle_out)
pickle_out.close()

clean_ham_bodies 	= clean_text(ham_bodies)
pickle_out = open("clean_ham_bodies.pickle", 'wb')
pickle.dump(clean_ham_bodies, pickle_out)
pickle_out.close()

clean_spam_subjects	= clean_text(spam_subjects)
pickle_out = open("clean_spam_subjects.pickle", 'wb')
pickle.dump(clean_spam_subjects, pickle_out)
pickle_out.close()

clean_spam_bodies 	= clean_text(spam_bodies)
pickle_out = open("clean_spam_bodies.pickle", 'wb')
pickle.dump(clean_spam_bodies, pickle_out)
pickle_out.close()