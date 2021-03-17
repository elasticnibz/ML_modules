# Importing generic libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats._continuous_distns import chi2
from sklearn.metrics import log_loss, auc, roc_curve, accuracy_score, silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Activation, Dropout
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from scipy.spatial.distance import cosine, cdist, pdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Categorical embeddings using keras
class CategoricalEmbeddings:
    def __init__(self, layers=[(50, "relu"), (15, "relu")], vecSize=None, epoch=50, batchSize=4):
        """
            Arguments:
                layers    = List([Int, Int]) NN layer sizes and activation func for training autoencoder
                            default = [(50, "relu"), (15, "relu")]
                vecSize   = [int] size of the embeddings
                            default = min(50, (inputSize+1)/2)
                epoch     = [int] number of epochs autoencoder to be trained
                batchSize = [int] size of training batches
        """
        self.epochs = epoch
        self.layers = layers
        self.vecSize = vecSize
        self.labelEncoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = Sequential()
        self.feature = None
        
    def fit(self, data, feature, featureList):
        """
            Arguments:
                feature     = [string] feature column to be vectorized
                featureList = List([strings]) target feature columns to be trained on
        """
        self.feature = feature
        inputSize = data[self.feature].unique().size

        if self.vecSize == None:
            embeddingSize = int(min(50, (inputSize+1)/2))
        else:
            embeddingSize = int(self.vecSize)
        print("Feature: {}\nInput Size: {}\nEmbedding Size: {}\n".format(self.feature, inputSize, embeddingSize))
        
        self.labelEncoder.fit(data[self.feature])
        self.scaler.fit(data[featureList].values.reshape(-1,len(featureList)))
        self.model.add(Embedding(input_dim=inputSize, output_dim=embeddingSize, input_length=1, name=self.feature+"_embedding"))
        self.model.add(Flatten())
        self.model.add(Dense(self.layers[0][0], activation=layers[0][1]))
        self.model.add(Dense(self.layers[1][0], activation=layers[1][1]))
        self.model.add(Dense(1))
        self.model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
        self.model.fit(x=self.labelEncoder.transform(data[self.feature].values.reshape(-1,1)), y=self.scaler.transform(data[featureList].values.reshape(-1,len(featureList))), epochs=self.epochs, batch_size=batchSize)
        return self

# Selecting heterogenous features for ML pipelines
# Example:
# p1 = Pipeline([("selector", TextFeatureSelector(column="words")), ("vectorizer", CountVectorizer())])
# p2 = Pipeline([("selector", NumFeatureSelector(column="numbers")), ("scaler", StandardScaler())])
# features = FeatureUnion([("words", p1), ("numbers", p2)])
class NumFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.key = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.key]]
    
class TextFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.key = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key]

class CatFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.key = column
        self.labelencoder = LabelEncoder()
        self.onehotencoder = OneHotEncoder()
    def fit(self, X, y=None):
        X1 = X[self.key].reset_index().drop(["index"], axis=1)
        X2 = pd.Series(self.labelencoder.fit_transform(X1))
        X_df = pd.DataFrame(X1,X2).reset_index().rename(columns={"index": "code"})
        self.onehotencoder.fit(X_df)
        return self
    def transform(self, X):
        X1 = X[self.key].reset_index().drop(["index"], axis=1)
        X2 = pd.Series(self.labelencoder.transform(X1))
        X_df = pd.DataFrame(X1,X2).reset_index().rename(columns={"index": "code"})
        arr = self.onehotencoder.transform(X_df).toarray()
        return arr
