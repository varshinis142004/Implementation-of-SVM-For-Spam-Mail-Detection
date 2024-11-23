# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook.

## Algorithm
1. Gather a labeled dataset containing both spam and non-spam emails, with labels typically as 0 (non-spam) and 1 (spam)
2. Clean the email text by removing punctuation, converting text to lowercase, and removing stop words to reduce noise.
3. Split the dataset into training and testing sets, typically using an 80-20 or 70-30 ratio.
4. Train the SVM on the training data, allowing it to learn patterns associated with spam and non-spam emails.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VARSHINI S
RegisterNumber: 212222220056
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
df=pd.read_csv("spam.csv",encoding='Windows-1252')

df.head()

df.info()

df.isnull().sum()

x=df['v2'].values
y=df['v1'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
#CountVectorizer is convert text into numerical data

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:
## head
![image](https://github.com/user-attachments/assets/d03a5251-c089-4134-8600-da9f4bfeb387)
## info
![image](https://github.com/user-attachments/assets/7d3c63fb-9cb2-4f1c-b5e5-e3654c5d52be)
## tail
![image](https://github.com/user-attachments/assets/fac01c0b-2895-4724-b663-c424822e1512)
## predicted value
![image](https://github.com/user-attachments/assets/7a25dac4-d201-4127-8b23-ef46456ca483)
## accuracy
![image](https://github.com/user-attachments/assets/64a669c7-3516-4ea1-b624-70b64440cf2b)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
