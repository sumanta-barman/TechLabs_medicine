#Imports
import pandas as pd
import tkinter as tk
from tkinter import *
from methods import * #class with functions

#import DF
df = pd.read_csv("sentiment_tweets1.csv")

#configuring DF
df.rename(columns={"message to examine":"message"}, inplace=True)

root = tk.Tk()
root.title("Depression Test")

canvas = tk.Canvas(root, width=300, height=400)
canvas.pack()

label = tk.Label(root,text="Hi! Enter your tweet here").pack()

tweet = tk.Entry(root, selectborderwidth=5)
tweet.pack()
tweet.insert(0,"")

result = Button(root, text="Send", command=lambda: process(root, df, tweet))
result.pack()

root.mainloop()