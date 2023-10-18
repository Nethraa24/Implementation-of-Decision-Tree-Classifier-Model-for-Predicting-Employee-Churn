# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: J.Nethraa
RegisterNumber:  212222100031
*/
import pandas as pd
data=pd.read_csv('/content/Employee.csv')

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data Head:
![image](https://github.com/Nethraa24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121215786/8850104f-05c5-44ac-af22-ad93fb88abe7)

### Dataset Info:
![image](https://github.com/Nethraa24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121215786/9b2b5660-3d11-4747-8244-35f4ff1b94a8)

### Null dataset:
![image](https://github.com/Nethraa24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121215786/40aea425-2092-4bed-852f-e618627784bc)

### Values Count in Left Column:
![image](https://github.com/Nethraa24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121215786/f9b639ea-ba36-4e3f-841c-16b4dce87b10)


### Dataset transformed head:
![image](https://github.com/Nethraa24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121215786/970953ab-f37c-4739-83e2-1c0d5cb0d8d9)

### x.head():
![image](https://github.com/Nethraa24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121215786/34760264-f038-4f2b-83ee-a56128efada6)

### Accuracy:
![image](https://github.com/Nethraa24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121215786/1d1a55c7-8c3e-46eb-a5a1-2b26eaeb5caa)

### Data Prediction:
![image](https://github.com/Nethraa24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121215786/0443eb09-e704-4d1d-bbdb-99dcdd722377)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
