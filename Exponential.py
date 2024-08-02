#importing MySql Connector
import mysql.connector

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
temperatures = df['temperature'].unique()

print(df.head())

def exp_decay(wavelength, a, b, c):
    return a * np.exp(-b * wavelength) + c

# Assiging the values to variables
X = df.iloc[:,1].values
y = df.iloc[:,-1].values

popt, pcov = curve_fit(exp_decay, X, y, p0=(1, 1e-3, 1))
colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(temperatures)))
plt.figure(figsize=(14, 8))
for i, temp in enumerate(temperatures):
    subset = df[df['temperature'] == temp]
    X = subset['wavelength'].values
    y = subset['radiation'].values 
    popt, pcov = curve_fit(exp_decay, X, y, p0=(1, 1e-3, 1))
    wavelength_range = np.linspace(X.min(), X.max(), 400)
    y_pred = exp_decay(wavelength_range, *popt)

    plt.scatter(X, y, color=colors[i], label=f'Observed Data {temp} K')
    plt.plot(wavelength_range, y_pred, color=colors[i], linestyle='--', label=f'Exp. Decay Fit {temp} K')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiation')
plt.title('Exponential Decay Fit of Radiation vs Wavelength with Temperature Effect')
plt.legend()
plt.show()
