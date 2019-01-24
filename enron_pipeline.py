"""
A streamlined approach to processing emails and classifying them as spam or not spam (ham)
This classification pipeline makes use of the simple preprocessing from the dataset authors, who have 
processed the raw Enron emails to produce files with a Subject on the first line, and the email content / body
on the remaining lines

"""
import pandas as pd
import os, time, pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 8)

# Read in 'cleaned' text from the processed emails
pickle_in 			= open("clean_ham_subjects.pickle", 'rb')
clean_ham_subjects 	= pickle.load(pickle_in)

pickle_in 			= open("clean_ham_bodies.pickle", 'rb')
clean_ham_bodies 	= pickle.load(pickle_in)

pickle_in 			= open("clean_spam_subjects.pickle", 'rb')
clean_spam_subjects 	= pickle.load(pickle_in)
pickle_in 			= open("clean_spam_bodies.pickle", 'rb')
clean_spam_bodies 	= pickle.load(pickle_in)

print("Files read")

# Create dataframes for each of the ham emails and spam emails using the same format with opposite labels
ham_df 		= pd.DataFrame({'Subject':clean_ham_subjects, 'Body':clean_ham_bodies, 'Spam':0})
spam_df 	= pd.DataFrame({'Subject':clean_spam_subjects, 'Body':clean_spam_bodies, 'Spam':1})

# Join all data samples together and shuffle 
data_df = pd.concat((spam_df, ham_df))
data_df = data_df.sample(frac=1, random_state=23).reset_index(drop=True)

print("Dataframe constructed")

# Add the message subject and body together, then create a feature vector for classification 
data_df['Space'] = ' '
Features = data_df['Subject'] + data_df['Space'] + data_df['Body']

# Use of the sklearn 'term frequency-inverse document frequency' vectorizer characterizes each email 
# by the importance of its words compared to the importance of said words within the entire corpus (email collection)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(Features)

X_train, X_test, y_train, y_test = train_test_split(features, data_df['Spam'], test_size=0.3, random_state=65)
print("Train/test split created")

# Create a Support Vector machine for Classification
# Manual testing found identical performance for several variations of the SVC parameters 
#	rbf and linear kernels performed identically, and the sigmoid kernel performed poorly without gamma tuning 
#	-> afterwards the sigmoid kernel performed slightly worse (1.5% less) than the default SVC()
svc = SVC()
svc.fit(X_train, y_train)
print("SVC fitted")

prediction = svc.predict(X_test)
print("Prediction accuracy: ", accuracy_score(y_test, prediction))