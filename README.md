# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
<img width="1017" height="403" alt="al ex2 ml" src="https://github.com/user-attachments/assets/a0edb177-3b70-4e4c-a919-d006e3795804" />

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K.GOUTHAM
RegisterNumber:  212223110019
*/
```
```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('/content/student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```
## Output:
<img width="617" height="728" alt="image" src="https://github.com/user-attachments/assets/043ef121-f44e-4d13-bbaa-ebefa6eedfa4" />
<img width="699" height="458" alt="image" src="https://github.com/user-attachments/assets/962f10c7-1057-491c-8ab6-0f022800df73" />
<img width="635" height="507" alt="image" src="https://github.com/user-attachments/assets/69d55f79-ed94-4de5-8dc0-aa8635610a85" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
