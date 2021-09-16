# TechLabs_medicine


# Description

We want to develop a machine learning model which will predict if a person may develop depression or not!

For more about git and github, you can watch this video...https://www.youtube.com/watch?v=RGOj5yH7evk&list=FLNqVzEKPAh5ntKuGcIQVYnw&index=1&t=1514s

For more about random forest, you can watch this video...https://www.youtube.com/watch?v=eM4uJ6XGnSM

For more about Sentiment Analysis with BERT, you can read this...https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

https://www.youtube.com/watch?v=oUpuABKoElw

or watch this video...https://www.youtube.com/watch?v=szczpgOEdXs

# Project steps
**Background**
Social media data provides valuable clues about physical and mental health conditions. Although the users may not aware with their health, analysing such valuable indicators help to predict the mental health status of the person. Studies have revealed that in addition to physical test, predictive screening methods can be used for number of mental health conditions like depression, addictions, post-traumatic stress disorder (PTSD), and suicidal tendency. Although considerable works are going on with this field, predictive health screeing with social media is still in starting stage. In this project, we tried to build state-of-art for predicting whether a person may have depression or not based on his/her tweets on the Twitter.

**Data Collection and Observation**
The tweets dataset stored on Kaggle was taken as dataset for study. 

**Summary Statistics**
Different information present on the dataset like number of rows, columns, variables, etc. were first observed. Number of tweets were counted along with their result like depressed or not were calculated and visualized using simple pie chart. 

**Machine Learining Models**
We trained supervised machine learning models to differentiate between depressed and healthy sample based upon their tweets. At the begining the sentences were passed to BERT models and pooling layer to generate their embeddings. 
The Classifiers were trained on a randomly selected 80% of total population, and tested on the remaining 20%. We basically used Random Forest Classifier and Naives Bayes Classifier and Support Vector Machine (SVM).
Out of several candidate algorithms, the SVM demonstrated the best performance (Accuracy of 99%). Moreover, Random Forest Classifier also showed good accuracy (97%).

Hyperparameter tuning with Random forest claffifier and found the better accuracy. 

