#importing MySql Connector
import mysql.connector

#Importing libraries
import numpy as np 
import pandas as pd
import  matplotlib.pyplot as plt

# Connecting with MySql Server 
conn=mysql.connector.connect(
    host="localhost",
    user="root",
    password="Anishakaaval35",
    database="blackbodydata"
)
# Query to select the table
query = "SELECT * FROM plancks;"

# Executing Query
cursor = conn.cursor()
cursor.execute(query)

# Fecthing all the rows
rows = cursor.fetchall()
# closing cursor and connection
cursor.close()
conn.close()


# Converting data from dataframe
df=pd.DataFrame(rows,columns=['id','wavelength','temperature','radiation'])

print(df.head())

# Assiging the values to variables
X = df.iloc[:,1].values
y = df.iloc[:,-1].values

# Splitting the data set into train_test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

X_train.shape
X_test.shape
y_train.shape
y_test.shape


# Reshaping the data set 
X_train = X_train.reshape(-1, 1) 
X_test = X_test.reshape(-1, 1) 

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Training the Linear Regression model to Training_set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting for Test_set
y_pred=regressor.predict(X_test)
print(np.set_printoptions(precision=2))
print(y_pred)

