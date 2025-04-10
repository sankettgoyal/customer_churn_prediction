# 🧠 Customer Churn Prediction using Machine Learning

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen.svg)

Predict telecom customer churn using various classification algorithms and real-world data. This machine learning project includes end-to-end steps from preprocessing to model evaluation, making it perfect for data science portfolios and case studies.

---

## 🚀 Project Overview

This project is focused on building machine learning models to predict whether a telecom customer is likely to churn (leave the service). It covers:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature selection and transformation
- Multiple machine learning model implementations
- Model evaluation and performance comparison

---

## 📂 Repository Contents

```
customer_churn_prediction/
│
├── Customer Churn Dataset.csv          # Dataset used for model training
├── customer-churn-prediction.ipynb     # Main Jupyter Notebook
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 📊 Dataset

This project uses the **Telco Customer Churn** dataset, available in the repository as:

📄 [`Customer Churn Dataset.csv`](Customer%20Churn%20Dataset.csv)

The dataset includes:
- Customer demographics (gender, senior citizen, etc.)
- Service usage (internet, phone service)
- Contract and billing information
- Target label: `Churn` (Yes/No)

> 📌 **Source**: [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 🔍 Machine Learning Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- CatBoost  

All models are evaluated using:
- Accuracy  
- Confusion Matrix  
- ROC-AUC Score  
- Precision, Recall, and F1 Score

---

## 🛠️ Tech Stack

**Languages & Tools:**  
`Python, Jupyter Notebook, Git, GitHub`

**Libraries:**  
`Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, CatBoost, Plotly, Missingno`

---

## 🖥️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/sankettgoyal/customer_churn_prediction.git
cd customer_churn_prediction

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Open the notebook
jupyter notebook customer-churn-prediction.ipynb
```

---

## ✅ Results Summary

- **Best Model:** Random Forest with high accuracy and ROC-AUC  
- **Key Features Influencing Churn:** Contract type, Monthly charges, Internet service  
- **Visualization:** Correlation matrix, KDE plots, bar graphs for feature analysis

---

## 📌 Possible Improvements

- Hyperparameter tuning with GridSearchCV
- Add SHAP for model explainability
- Build a simple Streamlit dashboard for predictions
- Automate model retraining on updated datasets

---

## 📬 Contact

**Sanket Goyal** 
🔗 [GitHub](https://github.com/sankettgoyal) | [LinkedIn](https://www.linkedin.com/in/sankettgoyal)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
