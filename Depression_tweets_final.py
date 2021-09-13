from wordcloud import WordCloud
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
#!pip install neattext
#!pip install plotly
import neattext.functions as nfx
import matplotlib.pyplot as plt
import plotly.express as plx
from sklearn.metrics import classification_report
#!pip install keras
import keras
from keras.layers import Embedding,Dense,LSTM,Bidirectional,GlobalMaxPooling1D,Input,Dropout
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.models import Sequential
#!pip install tensorflow
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("sentiment_tweets1.csv")

# preview the data
df.head()
df.info()
df.describe()
df.message
df['label (depression result)'].value_counts()
df['label (depression result)'].value_counts().index.values
#data set split

train_data,test_data=train_test_split(df, test_size=0.2, random_state=10)
train_data['label (depression result)'].value_counts().index.values
plt.figure(figsize=(10,8))
plt.pie(train_data['label (depression result)'].value_counts(),startangle=90,colors=['green', 'red'],
        autopct='%0.2f%%',labels=['no','yes'])
plt.title('Depression Or Not ?',fontdict={'size':20})
plt.show()
#['#06dddf','#000fbb'],
#Data cleaning
def clean_text(message):
    text_length=[]
    cleaned_text=[]
    for sent in tqdm(message):
        sent=sent.lower()
        sent=nfx.remove_special_characters(sent)
        sent=nfx.remove_stopwords(sent)
#         sent=nfx.remove_shortwords(sent)
        text_length.append(len(sent.split()))
        cleaned_text.append(sent)
    return cleaned_text,text_length
cleaned_train_text,train_text_length=clean_text(train_data.message)
cleaned_test_text,test_text_length=clean_text(test_data.message)

plt.figure(figsize=(20,12))
sns.distplot(train_text_length)
# plt.axis([-10,100,0,0.03])
plt.show()
tokenizer=Tokenizer()
tokenizer.fit_on_texts(cleaned_train_text)
word_freq=pd.DataFrame(tokenizer.word_counts.items(),columns=['word','count']).sort_values(by='count',ascending=False)
plt.figure(figsize=(20,20))
sns.barplot(x='count',y='word',data=word_freq.iloc[:100])
plt.show()
tweets = df.values[:,1]
labels = df.values[:,2].astype(float)
print (tweets[45], labels[45])
print (tweets[8005], labels[8005])
#!pip install sentence-transformers
from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = bert_model.encode(tweets, show_progress_bar=True)

print (embeddings.shape)


#embeddings = bert_model.encode(df.message, show_progress_bar=True)
#print (embeddings.shape)
embeddings
X = embeddings
y = df['label (depression result)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#20% better
from sklearn.ensemble import RandomForestClassifier
fit = RandomForestClassifier().fit(X_train, y_train)
print("The accuracy is : ")
np.sum(fit.predict(X_test)==y_test)/len(y_test)
words = np.array(['not depressed', 'depressed'])
print("The outcome is: ")
print(words[fit.predict([bert_model.encode("If you choose to define me by my mistakes, Remember redemption doesn't fall down at your feet.")])].squeeze())
print(words[fit.predict([bert_model.encode("the worst sadness is the sadness you've taught yourself to hide.")])].squeeze())
print(words[fit.predict([bert_model.encode("the best happiness is the sadness you've taught yourself to hide.")])].squeeze())