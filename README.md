# 🏠 House Price Prediction

A Machine Learning web application that predicts the estimated price of a house based on its property details such as area, number of bedrooms, bathrooms, stories, parking spaces, and house age.

The project uses **Python, Machine Learning, Pandas, Scikit-learn, Joblib, and Streamlit** to provide an interactive house price prediction system.

---

## 📌 Project Overview

House prices depend on several factors such as property size, number of rooms, location-related characteristics, and age of the house.

This project uses historical house data to train Machine Learning regression models and predict the estimated price of a new house.

The trained model is integrated into a **Streamlit web application**, where users can enter house details and instantly receive a predicted price.

---

## 🎯 Objectives

- Predict house prices using Machine Learning.
- Analyze the relationship between house features and price.
- Compare different regression algorithms.
- Select the best-performing Machine Learning model.
- Build an interactive web application using Streamlit.
- Provide a simple and user-friendly interface for prediction.

---

## ✨ Features

- 🏠 Modern house-themed web interface
- 🖼️ House image as website background
- 📐 Area input
- 🛏️ Bedroom input
- 🛁 Bathroom input
- 🏢 Number of stories
- 🚗 Parking spaces
- 📅 House age
- 🤖 Machine Learning-based prediction
- 💰 Estimated house price
- 📊 Displays the model used
- 🏡 Displays entered house information
- 🔄 Shows the complete ML workflow

---

## 🧠 Machine Learning

This is a **regression problem** because the model predicts a continuous numerical value: the house price.

The project can compare the following algorithms:

### 1. Linear Regression

Linear Regression finds a relationship between the input features and the house price.

### 2. Decision Tree Regression

Decision Tree Regression uses a tree-like structure to make predictions based on feature values.

### 3. Random Forest Regression

Random Forest combines multiple decision trees to improve prediction performance.

### 4. XGBoost

XGBoost is an optional gradient boosting algorithm that can also be used for regression.

The best-performing model is selected based on evaluation metrics.

---

## 📊 Input Features

The model uses the following features:

| Feature | Description |
|---|---|
| `area` | Area of the house in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of floors/stories |
| `parking` | Number of parking spaces |
| `age` | Age of the house in years |

### Target Variable

```text
price
