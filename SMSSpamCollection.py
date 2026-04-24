# MSA 8150 Machine Learning
# HW 1
# Jigar Patel


import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)


# Data reading

data = pd.read_csv("Machine_Learning\\SMSSpamCollection.txt", sep='\t', header=None, names=['label', 'message'])

print(len(data))

print(data.head())

# EDA

print(f"\nSpam percentage: {(data['label'] == 'spam').sum() / len(data) * 100:.2f}%")
print(f"Ham percentage: {(data['label'] == 'ham').sum() / len(data) * 100:.2f}%")

print(f"\nMissing values:\n{data.isnull().sum()}")

data['message_length'] = data['message'].apply(len)
data['word_count'] = data['message'].apply(lambda x: len(x.split()))

print(f"\nMessage length statistics:\n{data['message_length'].describe()}")
print(f"\nWord count statistics:\n{data['word_count'].describe()}")



stopwords = {'the', 'a', 'to', 'and', 'i', 'you', 'in', 'is', 'of', 'it', 
                 'for', 'on', 'my', 'that', 'me', 'have', 'your', 'with', 'im',
                 'do', 'just', 'will', 'can', 'get', 'so', 'are', 'but', 'was'}


spam_text = " ".join(data[data['label'] == 'spam']['message']).lower()
spam_text = re.sub(r'[^\w\s]', '', spam_text)
spam_words = [word for word in spam_text.split() if word not in stopwords and len(word) > 2]
spam_words = Counter(spam_words)

ham_text = " ".join(data[data['label'] == 'ham']['message']).lower()
ham_text = re.sub(r'[^\w\s]', '', ham_text)
ham_words = [word for word in ham_text.split() if word not in stopwords and len(word) > 2]
ham_words = Counter(ham_words)

print("\nTop 10 words in spam messages:")
for word, count in spam_words.most_common(10):
    print(f"  {word:15s} : {count:4d}")

print("\nTop 10 words in ham messages:")
for word, count in ham_words.most_common(10):
    print(f"  {word:15s} : {count:4d}")


sns.histplot(data=data, x='message_length', hue='label', bins=50, kde=True)
plt.title('Message Length Distribution by Label')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.show()

sns.histplot(data=data, x='word_count', hue='label', bins=50, kde=True)
plt.title('Word Count Distribution by Label')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# Data cleaning and preprocessing

data['message'] = data['message'].str.lower()
data['message'] = data['message'].str.replace(r'[^\w\s]', '', regex=True)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training and evaluation

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}
results = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }  
    print(f"\n{name} Results:")
    print(f"Accuracy: {results[name]['accuracy']:.4f}")
    print(f"Precision: {results[name]['precision']:.4f}")
    print(f"Recall: {results[name]['recall']:.4f}")
    print(f"F1 Score: {results[name]['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(results[name]['confusion_matrix'])
    print("Classification Report:")
    print(results[name]['classification_report'])

