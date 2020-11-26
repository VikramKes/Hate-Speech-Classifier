import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
#%matplotlib inline

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

nltk.download('stopwords')
nltk.download('wordnet')
#stopwords.words('english')

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def load_data(train_data_path,test_data_path):
    return (pd.read_csv(train_data_path),pd.read_csv(test_data_path))


def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


def preprocessing(data,data_test):
    #convert to string type
    data['text'] = data['text'].astype(str)
    data_test['text'] = data_test['text'].astype(str)
    #convert to the lowercase
    data["text"] = data["text"].str.lower()
    data_test["text"] = data_test["text"].str.lower()
    #removing hyperlinks
    data['text'] = data['text'].str.replace('http\S+|www.\S+', '', case=False)
    data_test['text'] = data_test['text'].str.replace('http\S+|www.\S+', '', case=False)
    #remove punctuations
    data["text"] = data["text"].apply(lambda text: remove_punctuation(text))
    data_test["text"] = data_test["text"].apply(lambda text: remove_punctuation(text))
    #remove emoji's
    data["text"] = data["text"].apply(lambda text: remove_emoji(text))
    data_test["text"] = data_test["text"].apply(lambda text: remove_emoji(text))
    #stopwords removal
    data["text"] = data["text"].apply(lambda text: remove_stopwords(text))
    data_test["text"] = data_test["text"].apply(lambda text: remove_stopwords(text))
    #Remove Numbers
    data['text'] = data['text'].str.replace('\d+', '')
    data_test['text'] = data_test['text'].str.replace('\d+', '')
    #stemming
    data["text"] = data["text"].apply(lambda text: stem_words(text))
    data_test["text"] = data_test["text"].apply(lambda text: stem_words(text))
    return data,data_test
    

def vectorize_and_transform(data,data_test):
    wordVector=CountVectorizer()
    bagOfWords=wordVector.fit_transform(data['text'])
    bagOfWords_1=wordVector.transform(data_test['text'])
    objects=TfidfTransformer()
    featureData=objects.fit_transform(bagOfWords)
    featureData_test=objects.transform(bagOfWords_1)
    return featureData,featureData_test

def train_test_split_data(featureData,data):
    X_train,X_test,Y_train,Y_test=train_test_split(featureData,data['labels'],test_size=0.20,random_state=3)
    return X_train,X_test,Y_train,Y_test
    
    
def prediction(X_train,Y_train):
    model2=LogisticRegression()
    model2.fit(X_train,Y_train)
    return model2



data_train,data_test=load_data('train.csv','test.csv')
data_train,data_test=preprocessing(data_train,data_test)
featureData,featureData_test=vectorize_and_transform(data_train,data_test)
X_train,X_test,Y_train,Y_test=train_test_split(featureData,data_train['labels'])
model2=prediction(X_train,Y_train)
print(accuracy_score(Y_test,model2.predict(X_test)))
print(f1_score(Y_test, model2.predict(X_test)))