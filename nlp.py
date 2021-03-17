# Importing generic libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats._continuous_distns import chi2
from sklearn.metrics import log_loss, auc, roc_curve, accuracy_score, silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Activation, Dropout
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from scipy.spatial.distance import cosine, cdist, pdist
from sklearn.cluster import KMeans, AgglomerativeClustering

# Removal of punctuations from text
def strip_punctuations(text):
    stripped = " ".join(" ".join(["" if ch in string.punctuation else ch for ch in text]).split())
    return stripped

# Tokenize text into words
def tokenize(text):
    tokens = [word for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    tokens = [word.lower() for word in tokens]
    return tokens

# Removing stopwords
def remove_stopwords(tokens):
    stopwds = stopwords.words("english")
    tokens_ = [token for token in tokens if token not in stopwds]
    return tokens_

# Filter out short words
def filter_short_words(tokens, length=3):
    tokens_ = [word for word in tokens if len(word)>=length]
    return tokens_

# Stemming words down to the root
def stem_words(tokens):
    stemmer = PorterStemmer()
    tokens_ = [stemmer.stem(word) for word in tokens]
    return tokens_

# POS-tagging (required for lemmatization)
def tag_pos(tokens):
    """
        This returns part of speech. N, V, A, R.
    """
    tagged_corpus = pos_tag(tokens)
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    simplified_tags = [(word, tag_dict[tag[0]]) for (word, tag) in tagged_corpus]
    return simplified_tags

# Lemmatize words
def lemmatize(tagged_corpus):
    lemmatizer = WordNetLemmatizer()
    lemmatized = " ".join([lemmatizer.lemmatize(token, tag) for (token, tag) in tagged_corpus])
    return lemmatized

# Text preprocessing pipeline
def process_text(text):
    step1 = strip_punctuations(text)
    step2 = tokenize(step1)
    step3 = remove_stopwords(step2)
    step4 = filter_short_words(step3)
    step5 = stem_words(step4)
    step6 = tag_pos(step5)
    step7 = lemmatize(step6)
    return step7