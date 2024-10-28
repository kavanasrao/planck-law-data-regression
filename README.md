# Planck's Radiation Simulation and Data Analysis
This project simulates black body radiation based on Planck’s Law and performs linear regression analysis to study the relationship between wavelength and radiation at varying temperatures. The data is stored in MySQL for structured analysis and visualization.

## 📑 Table of Contents
- Overview
- Features
- Tech Stack
- Getting Started
- Project Structure
- Usage
- Results

## Overview
Planck's Law is a fundamental principle in physics that describes the radiation emitted by a black body in thermal equilibrium at a given temperature. This project applies Planck’s radiation formula to simulate radiation intensities at various wavelengths and temperatures, storing the results in a MySQL database, and then analyzing the data using machine learning techniques.

## Features
Calculation of black body radiation at multiple temperatures and wavelengths using Planck’s Law
Storage of calculated data in a MySQL database
Data extraction, preprocessing, and analysis with linear regression to explore relationships
Visualization of the training and test sets, residuals, and prediction accuracy

## Tech Stack
- Python for calculations, data handling, and modeling
- MySQL for data storage and retrieval
- NumPy and Pandas for numerical operations and data manipulation
- Matplotlib for visualizing radiation data and regression analysis
- scikit-learn for machine learning model implementation

## Getting Started
Prerequisites
Ensure you have the following installed:

Python 3.x
MySQL Server
Python Libraries: numpy, pandas, mysql-connector-python, matplotlib, scikit-learn

## Installation
- Clone the repository:
git clone https://github.com/yourusername/PlancksRadiation.git
cd PlancksRadiation

- Install required Python libraries:
pip install numpy pandas mysql-connector-python matplotlib scikit-learn

- Set up a MySQL database and table for storing data:
CREATE DATABASE blackbodydata;
USE blackbodydata;
CREATE TABLE plancks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    wavelength_nm FLOAT,
    temperature INT,
    radiation FLOAT
);
- Update the MySQL credentials in the code (user, password, database) to match your setup.

## Project Structure
plancks_radiation.py: Main script for calculating radiation, storing data, and performing regression analysis.
screenshots: Contains images of output results (optional folder to add)

## Usage
Run the Script:
python plancks_radiation.py

## The code performs the following:

- Calculates radiation using Planck's formula for a range of temperatures and wavelengths.
- Stores calculated data in MySQL.
- Retrieves and analyzes data using linear regression.
- Visualizes training/test data, residuals, and regression accuracy.


## Results
This project provides insights into the black body radiation curve and demonstrates a relationship between wavelength and radiation intensity. Results are visualized through scatter plots, regression lines, and residual plots for analysis.

## Evaluation Metrics
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
