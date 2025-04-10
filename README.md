# 🧠 Customer Churn Prediction Using Machine Learning

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen.svg)

Predict customer churn with machine learning using telecom user data. This project identifies which customers are likely to leave a service provider, helping businesses improve customer retention. Perfect for your data science portfolio and real-world business case scenarios.

---

## 🚀 Project Overview

Customer churn prediction helps businesses understand why users leave and what patterns lead to it. This repository demonstrates:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation using multiple algorithms
- Performance comparison & feature importance

---

## 📊 Dataset

The dataset includes customer demographics, service usage, and billing data, with a target variable `Churn`.

> 📂 **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 📈 Algorithms Used

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Gradient Boosting (XGBoost, CatBoost)

All models are evaluated using metrics like:
- Accuracy  
- Precision, Recall, F1 Score  
- ROC-AUC Score  
- Confusion Matrix  

---

## 🛠️ Tech Stack

**Languages & Tools**:  
`Python, Jupyter Notebook, Git, GitHub`

**Libraries**:  
`Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, CatBoost, Plotly, Missingno`

---

## 📂 Repository Structure

```
customer_churn_prediction/
│
├── customer-churn-prediction.ipynb     # Main analysis notebook
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation
└── .gitignore
```

---

## 🖥️ How to Run

```bash
# Clone the repository
git clone https://github.com/sankettgoyal/customer_churn_prediction.git
cd customer_churn_prediction

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook customer-churn-prediction.ipynb
```

---

## ✅ Results & Insights

- Best performing model: **Random Forest** with ~84% accuracy  
- Key churn indicators: Contract Type, Monthly Charges, Internet Service  
- Visual insights via heatmaps, histograms, and ROC curves  

---

## 📌 Future Scope

- Hyperparameter tuning using GridSearchCV  
- Model deployment using Streamlit or Flask  
- Automate predictions on new incoming data  
- Integrate SHAP or LIME for explainability  

---

## 📬 Connect With Me

**Sanket Goyal** 
🔗 [GitHub](https://github.com/sankettgoyal) | [LinkedIn](https://www.linkedin.com/in/sankettgoyal)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
