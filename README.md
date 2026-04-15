# 📉 Telecom Customer Churn Prediction – Basic Level

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Project Overview

Customer churn is a major business challenge in the telecom industry — losing a customer costs
significantly more than retaining one. This project builds a complete churn prediction pipeline:
from raw data exploration to a deployed Streamlit app that predicts churn in real time.


---

## 🎯 Business Problem

A telecom company is experiencing revenue loss due to customer attrition. The goals are:

- Identify customers who are likely to churn
- Understand the key drivers behind churn behavior
- Predict churn early enough to support targeted retention strategies

---

## 📂 Dataset

| Property | Details |
|---|---|
| Source | Telecom Customer Churn Dataset (Kaggle) |
| Size | ~7,000 rows · 21 features |
| Target | `Churn` — Yes / No (binary classification) |
| Key features | Tenure, contract type, monthly charges, internet service, payment method |

> **Class imbalance note:** The dataset is imbalanced (~26% churn). This was handled using
> SMOTE during model training.

---

## 🔍 Project Workflow

1. Data Understanding
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Preparation
5. Model Building
6. Model Evaluation
7. Business Insights & Recommendations

---

## 🛠️ Tools & Technologies

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn, imbalanced-learn (SMOTE) |
| App | Streamlit |
| Environment | Jupyter Notebook |

---

## 📈 Key Insights

- Month-to-month contract customers churn significantly more than long-term contract holders
- Customers with higher monthly charges and no online security are at greater churn risk
- Short-tenure customers (< 12 months) represent the highest churn segment

---

## 📊 Model Used

**Logistic Regression**

| Metric | Score |
|---|---|
| Accuracy | 0.7402 |
| Recall | 0.8128 |
| F1-Score | 0.6242 |

Evaluation via Confusion Matrix, Accuracy, Recall, and F1-Score.

---

## 💡 Business Recommendations

- Offer loyalty discounts to customers in their first year to reduce early churn
- Promote long-term contracts with incentives for month-to-month users
- Bundle security and support services to improve perceived value

---

## 🚀 Future Work

This project will be extended into **Intermediate** and **Advanced** levels with:
- More complex models (Random Forest, XGBoost)
- Advanced feature engineering
- Business ROI analysis

---

## 👩‍💻 Author

**Kushalini Sreeja** — CSE (Data Science), Raghu Engineering College  
[github.com/kushalinisreeja](https://github.com/kushalinisreeja)
