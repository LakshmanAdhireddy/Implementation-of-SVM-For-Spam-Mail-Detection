# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1 :
Import the necessary python packages using import statements.

Step 2 :
Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

Step 3 :
Split the dataset using train_test_split.

Step 4 :
Calculate Y_Pred and accuracy.

Step 5 :
Print all the outputs.

Step 6 :
End the Program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Lakshman
RegisterNumber:  212222240001
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["EmailText"].values
y=data["Label"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()

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

### DATA.HEAD() :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707265/d0d388e4-2ce8-4927-8d9e-94f8ea5893fd)

### DATA.INFO() :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707265/13fa1c74-acde-44c5-baf6-a2d0110f5f3e)

### DATA.ISNULL().SUM() :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707265/ad71df9f-a543-4ab6-9a97-70f7c5d7d168)

### Y_PRED :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707265/8e1b07b5-2a42-4e4b-9bba-9e3fd894c57a)

### ACCURACY :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707265/d400a286-0333-48bb-8d15-030edff6c835)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
