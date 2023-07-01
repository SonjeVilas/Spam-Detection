# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df= pd.read_csv('SMSSpamCollection', sep='\t', header=None)
df.columns = [ 'class', 'message']

cv=CountVectorizer()
data=cv.fit_transform(df['message'])
data=data.toarray()
data=pd.DataFrame(data)

#Splitting Training and Test Set
target=df['class']
x_train,x_test,y_train,y_test=train_test_split(data,target)

from sklearn.naive_bayes import MultinomialNB
nv=MultinomialNB()

#Fitting model with trainig data
nv.fit(x_train, y_train)

pickle.dump(nv, open('model.pkl','wb'))
new_messeges = ["free free absolute free register today demo",]
test=cv.transform(new_messeges).toarray()

model = pickle.load(open('model.pkl','rb'))
print(model.predict(test))