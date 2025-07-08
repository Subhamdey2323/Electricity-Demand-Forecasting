# ⚡ Energy Load Forecasting using Machine Learning

## 📌 Project Objective
The goal of this project is to forecast the **total energy load** using historical weather and generation features. This helps optimize energy distribution and improve grid reliability.

---

## 📂 Dataset
- Source: [Insert dataset link, e.g., Kaggle or UCI]
- Contains timestamps, weather data, energy generation by source, and actual load demand.

---

## 🧠 Models Used
- Linear Regression
- Lasso / Ridge Regression
- K-Neighbors Regressor
- Decision Tree
- Random Forest
- **XGBoost (Best)**
- AdaBoost

---

## 🛠️ Preprocessing & Feature Engineering
- Converted date-time to `month`, `week`, `day`, `hour`
- Created `shifted_demand_supply` to reflect lag
- One-hot encoded categorical features
- Standardized numerical columns

---

## 📊 Model Evaluation

| Metric | XGBoost (Best Model) |
|--------|----------------------|
| R²     | 0.9104               |
| MAE    | 566.38               |
| RMSE   | 768.47               |

Underprediction and overprediction MAE were also calculated to understand bias.

---

## 📈 Visuals
- Actual vs Predicted Scatter Plot
- Prediction Error Histogram
- Model Comparison Chart

---

## 📦 File Highlights
- `final_model_analysis.ipynb` – Full notebook with EDA, modeling, and evaluation
- `XGB_Prediction_Errors.csv` – Error comparison table
- `images/` – Model visuals

---

## 🚀 Future Improvements
- Hyperparameter tuning via GridSearchCV
- Cross-validation for more robustness
- SHAP for model explainability
- Deploy using Streamlit or Power BI dashboard

---

## 👨‍💻 Author
- **Subham Dey**
- GitHub: [https://github.com/Subhamdey2323]
- LinkedIn: [www.linkedin.com/in/subhamdey]

---

⭐️ Don't forget to star this repo if you found it helpful!
