#importing MySql Connector
import mysql.connector

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

print(X)
print(y)

# Splitting the data set into train_test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Reshaping the data set 
X_train = X_train.reshape(-1, 1) 
X_test = X_test.reshape(-1, 1) 

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Training the Linear Regression model to Training_set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting for Test_set
y_pred=regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X,y,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='Blue')
plt.title("Black Body Radiation(Planck's Law)")
plt.xlabel("Wavelenth_nm")
plt.ylabel("Radiaton")
plt.show()

# Visualising the Test set results
plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='Blue')
plt.title('Black Body Radition(test set)')
plt.xlabel("Wavelenth_nm")
plt.ylabel("Radition")
plt.show()

# Evaluating Model with Different Metrics
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print(r2_score(y_pred,y_test))
print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='w', s=70)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Wavelength')
plt.ylabel('Radiation')
plt.title('Planck/"s radition ')
plt.show()

# Residual plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='red', edgecolor='w', s=70)
plt.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), colors='black', linestyles='dashed')
plt.xlabel('Wavelength')
plt.ylabel('Radiation')
plt.title(' Planck/"s radition')
plt.show()

# Coefficient plot
#coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
#print(coefficients)