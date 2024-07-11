# planck's Black Body radition
import math
import numpy as np
# Connecting to Mysql 
import mysql.connector

# Planck's Law Constants
h=6.62607015e-34
c=299792458
k=1.380649e-23

# placks_formula
def plancks_radiation(wavelength,temperature):
    numerator=(2*h*(c**2))/(wavelength**5)
    denominator=(math.exp(h*c/(wavelength*k*temperature))-1)
    return numerator/denominator

# creating values for Wavlength
wavelengths= np.linspace(100e-9, 5000e-9, 100)
# For specific Temperature
temperature=[3000,4000,5000,6000]

data=[]
# Looping through different Wavlength & Temperature
for T in temperature:
    for wavelength in wavelengths:
        radiation=plancks_radiation(wavelength,T)
        data.append((float(wavelength * 1e9), int(T), float(radiation)))

# Connecting to the database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="***************",
    database="blackbodydata"
)
cursor = conn.cursor()

insert_query = """
INSERT INTO plancks (wavelength_nm, temperature, radiation)
VALUES (%s, %s, %s)
"""

for entry in data:
    wavelength, temp, radiance = entry
    cursor.execute(insert_query, (wavelength, temp, radiance))

conn.commit()
cursor.close()
conn.close()
