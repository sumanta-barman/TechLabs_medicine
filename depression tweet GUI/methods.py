from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
from tkinter import *

#transform tweets to embeddings
def bert(df):
  bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
  embeddings = bert_model.encode(df.values[:,1], show_progress_bar=True)
  X = embeddings
  y = df.values[:,2]
  return X,y, bert_model

#data split
def split(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                          test_size=0.2, random_state=42)
  y_train = y_train.astype(int)
  y_test = y_test.astype(int)
  return X_train, X_test, y_train, y_test

#process and analyse the df and tweet
def process(root, tweet, bert_model, X, y):
    X_train, X_test, y_train, y_test = split(X,y)
    RF = RandomForestClassifier().fit(X_train, y_train)
    words = np.array(['not depressed', 'depressed'])
    pred= RF.predict([bert_model.encode(tweet)])
    Label(root, text= words[pred].squeeze()).pack()
    
