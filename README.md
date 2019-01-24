# Email Spam Classifier
An implementation of a sklearn SVC classifier to label emails from the Enron email corpus as spam or ham

This process of classifying emails into spam or ham has been implemented in two steps. The first is to process the emails into lists of words that are suitable to analyze (words with lower value in regards to the email's intent are removed), and then a second step characterizes the emails based on the words within each email and how those words relate to the words within the entire email corpus (using a TFIFD vectorizer).
