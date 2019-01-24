# Email Spam Classifier
An implementation of a sklearn SVC classifier to label emails from the Enron email corpus as spam or ham (legitimate messages).

This process of classifying emails into spam or ham has been implemented in two steps. The first is to process the emails into lists of words that are suitable to analyze (words with lower value in regards to the email's intent are removed), followed by characterizing the emails based on those words (and how they relate to the entire corpus) and subsequently training a Support Vector machine for Classification (SVC). The SVC is trained on 23,601 samples (51%/49% ham/spam split) and tested on 10,115 samples (51%/49% ham/spam split), achieving an f1 score of 98.97%.

This classifier makes use of an Enron email dataset by V. Metsis, I. Androutsopoulos and G. Paliouras, and can be accessed [here](http://www2.aueb.gr/users/ion/data/enron-spam/index.html). 

## Preparing the emails
The emails from this dataset have been split into 6 chunks of ham (16545 total) and spam (17171 total), and have been reduced to text files with a subject on the first line and the message text on the remaining lines. 

The first step to preparing the emails is to separate the email subject and body into two separate lists of words. The two are separated to facilitate potential future improvements where more emphasis is placed on the subjects, or to test the ability of the classifier based soley on the email subject. The latter case could provide a model that would train and inference much faster. 

The second step is to reduce the subjects and bodies to suitable words to be analyzed by the classifier. This is done by removing words that are common in both contexts and are not likely to help us differentiate between spam or legitimate messages, too short to provide any useful information, or aspects of emails too specific to provide statistical insight (web addresses, contact addresses). The 'clean_text()' function within 'email_utils.py' accomplishes this, which is called within the 'process_emails.py' script. This script goes on to save the 4 files (subjects and bodies for ham and spam emails) into serialized pickle files so that the emails don't need to be processed everytime the classifier script is run.

## Creating features and creating a classifier
The 'email_pipeline.py' script is designed to load in processed emails, translate the input data into feature vectors, and build a high performing classifier. The script does this efficiently by making use of many aspects of the scikit-learn library, primarily the TfidfVectorizer and the SVC classifier. The term frequency–inverse document frequency (TFIDF) statistic "(is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.)[https://en.wikipedia.org/wiki/Tf%E2%80%93idf]" This statistic shows importance to words which are common in an email but less common to the overall corpus. This highlights words that can prove very valuable in indicating the intent of the email (ham vs spam). Once a feature vector is created for the input data, it is split into train and test groups. 
