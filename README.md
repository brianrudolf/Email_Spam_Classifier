# Email Spam Classifier
An implementation of a sklearn SVC classifier to label emails from the Enron email corpus as spam or ham (legitimate messages).

This process of classifying emails into spam or ham has been implemented in two steps. The first is to process the emails into lists of words that are suitable to analyze (words with lower value in regards to the email's intent are removed), followed by characterizing the emails based on those words (and how they relate to the entire corpus) and subsequently training a Support Vector machine for Classification (SVC). The SVC is trained on xxx samples (split) and tested on yyy samples (split) and achieves an accuracy metric of zzz%.

This classifier makes use of an Enron email dataset by V. Metsis, I. Androutsopoulos and G. Paliouras, and can be accessed [here](http://www2.aueb.gr/users/ion/data/enron-spam/index.html). 

## Preparing the emails
The emails from this dataset have been split into 6 chunks of ham (16545 total) and spam (17171 total), and have been reduced to text files with a subject on the first line and the message text on the remaining lines. 

The first step to preparing the emails is to separate the email subject and body into two separate lists of words. The two are separated to facilitate potential future improvements where more emphasis is placed on the subjects, or to test the ability of the classifier based soley on the email subject. The latter case could provide a model that would train and inference much faster. 

The second step is to reduce the subjects and bodies to suitable words to be analyzed by the classifier. This is done by removing words that are common in both contexts and are not likely to help us differentiate between spam or legitimate messages, too short to provide any useful information, or aspects of emails too specific to provide statistical insight (web addresses, contact addresses). The 'clean_text()' function within 'email_utils.py' accomplishes this.

## Creating features

within each email and how those words relate to the words within the entire email corpus (using a TFIFD vectorizer).
