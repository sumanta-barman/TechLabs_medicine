#Imports
import pandas as pd
import tkinter as tk
from methods import * #class with functions

#import DF
df = pd.read_csv("sentiment_tweets1.csv")

#configuring DF
df.rename(columns={"message to examine":"message"}, inplace=True)

#preparing bert
X,y,bert_model = bert(df)

root = tk.Tk()
root.title("Depression Test")

label = tk.Label(root,text="Hi! This program try to predic if you probably are or not depressed", width=100).grid(ipadx=50, ipady=50)

tweet = tk.Entry(root, selectborderwidth=5, text="Enter your tweet here", insertwidth="100").grid(ipadx=50, ipady=50)
tweet = tweet.get()

result = Button(root, text="Send", command=lambda: process(root, tweet, bert_model, X, y))
result.grid()

root.mainloop()