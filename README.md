🚀 End-to-End Automated Machine Learning (AutoML) Platform

**A Multi-Page Streamlit Application for EDA → Cleaning → Model Training → Evaluation → Explainability → Report Generation**

 📌 **Overview**

This project is an **End-to-End AutoML Platform** built using **Streamlit, Scikit-Learn, SHAP, and Plotly**.
It allows users—students, beginners, data scientists—to upload a dataset and automatically perform:

* 📊 **Exploratory Data Analysis (EDA)**
* 🧹 **Automated Data Cleaning**
* 🎯 **Target Selection & Task Detection**
* ⚙️ **Model Training (Regression, Binary, Multi-Class)**
* 🏆 **Leaderboard with CV & Test Scores**
* 📈 **Model Evaluation (Confusion Matrix, ROC, PR Curve, Residuals, etc.)**
* 🔍 **SHAP Explainability (Feature Importance & Local Explanations)**
* 🧪 **Predict on New Data**
* 📄 **Auto-Generated PDF Report**
* 💻 **Model Code Export (Python Script + Deployment Template)**

This project replicates features similar to **Google AutoML / Azure AutoML / H2O AutoML**, but in a simple academic-friendly UI.


🌟 **Key Features**

**1️⃣ EDA + Cleaning**

* Missing value analysis
* Data types summary
* Correlation heatmap
* Missingness matrix
* Automatic cleaning (drop high-missing columns, impute, fix categorical issues)



**2️⃣ Smart Model Training**

Supports:

🔹 **Regression Models**

* Linear Regression
* Ridge / Lasso / ElasticNet
* SVR
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting
* KNN
* XGBoost

🔹 **Classification Models**

* Logistic Regression
* Random Forest
* Gradient Boosting
* KNN
* SVC
* XGBoost
* Optional **SMOTE handling for imbalance**


**3️⃣ Automatic Leaderboard**

Shows **CV scores** and **Test scores** for all models.
Highlights the **best model** automatically.



**4️⃣ Model Explainability**

Includes:

* Global SHAP summary
* SHAP feature importance
* SHAP dependence plots
* Waterfall plot
* Force plot (HTML interactive)

👉 Automatically warns the user if SHAP is too slow for high-dimensional datasets.



**5️⃣ Prediction Page**

* Download model input template
* Upload new data
* Predict with best model
* Shows class probabilities for classification models


**6️⃣ PDF Report Generator**

Generates professional PDF with:

* Model scores
* Evaluation charts
* SHAP results
* Steps performed

Perfect for **academic submissions, hackathons, and interviews**.



 **7️⃣ Model Export (Deployment Code Generator)**

Exports:

* Python training script template
* Prediction script
* Requirements.txt
* README + Documentation
* Jupyter Notebook (optional)

👉 Includes disclaimers and TODOs like real AutoML systems (Google, Azure).



 🧱 **Project Structure**

```
auto-ml-report/
│
├── app.py
├── pages/
│   ├── 1_EDA_and_Cleaning.py
│   ├── 2_Train_Models.py
│   ├── 3_Evaluate_and_Explain.py
│   ├── 4_Predict_New_Data.py
│   ├── 5_Generate_Report.py
│   ├── 6_Model_Code.py
│
├── engines/
│   ├── regression_engine.py
│   ├── classification_engine.py
│
├── utils/
│   ├── eda.py
│   ├── evaluation.py
│   ├── shap_engine.py
│   ├── report_generator.py
│
├── reports/
│   ├── eda/
│   ├── shap/
│   ├── pdf/
│
└── requirements.txt
```

---

 🛠️ **Tech Stack**

| Layer          | Tools                       |
| -------------- | --------------------------- |
| Frontend       | Streamlit                   |
| ML Models      | Scikit-Learn, XGBoost       |
| Explainability | SHAP                        |
| Visualization  | Plotly, Matplotlib, Seaborn |
| Backend Logic  | Python                      |
| Reporting      | ReportLab                   |

---

💻 **How to Run Locally**

1. Clone the repository

```
git clone https://github.com/<your-username>/AutoML-project.git
cd AutoML-project
```

2. Create virtual environment

```
python -m venv venv
```

3. Activate

 Windows:

```
venv\Scripts\activate
```

 Mac/Linux:

```
source venv/bin/activate
```

 4. Install dependencies

```
pip install -r requirements.txt
```

5. Run Streamlit app

```
streamlit run app.py
```



🏆 **Why This Project Stands Out**

Unlike simple Streamlit ML apps, this project includes:

✔ Full pipeline automation
✔ Multi-page UI
✔ SHAP explainability
✔ PDF reporting
✔ Code generation like Google AutoML
✔ Deployment-ready model export
✔ Professional-grade architecture



👨‍💻 **Author**
**M. V. G. N. Manikanta Chitimereddi**
B.Tech | Machine Learning Enthusiast
GitHub: https://github.com/manikanta2027
LinkedIn: www.linkedin.com/in/manikantachitimereddi

