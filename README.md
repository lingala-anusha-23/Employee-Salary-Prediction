
# 📊 Employee Salary Prediction

## 🌟 Overview
This project focuses on predicting employee salaries 💰 based on various features such as age, education, occupation, and more. The dataset includes demographic details, work history, and income levels.

## 📂 Dataset
The dataset includes the following columns:
- **`age`** 🎂: Age of the employee
- **`workclass`** 🏢: Type of employment (e.g., Private, Self-emp-not-inc)
- **`fnlwgt`** ⚖️: Final weight
- **`education`** 🎓: Highest level of education
- **`educational-num`** 🔢: Numerical representation of education level
- **`marital-status`** 💍: Marital status
- **`occupation`** 👨‍💼: Job role
- **`relationship`** 👨‍👩‍👧‍👦: Relationship status
- **`race`** 🌍: Ethnicity
- **`gender`** ♀️♂️: Gender
- **`capital-gain`** 📈: Capital gains
- **`capital-loss`** 📉: Capital losses
- **`hours-per-week`** ⏳: Hours worked per week
- **`native-country`** 🌎: Country of origin
- **`income`** 💵: Income level (<=50K or >50K)

## 🧹 Data Preprocessing
1. **Handling Missing Values** ❓: Missing values in `workclass` and `occupation` were replaced with "Others".
2. **Outlier Detection** 📊: Outliers in `age`, `capital-gain`, and `educational-num` were removed.
3. **Data Cleaning** 🧼: Irrelevant categories in `workclass` (e.g., "Without-pay", "Never-worked") were removed.

## 🔍 Exploratory Data Analysis (EDA)
- 📦 **Box plots** were used to detect and remove outliers.
- 📊 **Basic statistics** and value counts were analyzed to understand feature distributions.

## 🚀 Features

- **📝 Interactive Input**: Adjust employee details using sidebar sliders and dropdowns
- **🔮 Prediction**: Instantly predicts salary class with confidence percentage
- **📊 Feature Importance**: Visualizes key decision factors (when available)
- **📱 Responsive Design**: Works beautifully on all devices

## 🚀 Next Steps
1. **Feature Engineering** 🛠️: Create new features or transform existing ones.
2. **Model Training** 🤖: Train ML models to predict income levels.
3. **Evaluation** 📏: Assess model performance using metrics like accuracy, precision, and recall.

## ⚙️ Dependencies
- **Python 3.x** 🐍
- **pandas** 🐼
- **matplotlib** 📊
- **scikit-learn** (for future model development) 🔧

## 🏃‍♂️ Usage
1. **Clone** the repository.  
   ```bash
   git clone <repo-url>
   ```
2. **Install dependencies**.  
   ```bash
   pip install pandas matplotlib scikit-learn
   ```
3. **Run the Jupyter notebook** to preprocess data and perform EDA.  
4. **Train and evaluate models** for salary prediction.  

---

✨ Predict whether an employee earns >$50K or ≤$50K based on demographic and employment factors.


## 🎮 How to Use

1. **🛠️ Adjust Inputs** in the sidebar:
   - 👨‍💼 Demographic info (age, gender, race)
   - 💼 Employment details (occupation, work hours)
   - 💰 Financials (capital gain/loss)
   - 🎓 Education & relationships

2. **🚀 Click** the "Predict Salary Class" button

3. **📈 View Results**:
   - ✅ Prediction (>50K or ≤50K)
   - 💯 Confidence percentage
   - 🔍 Feature importance chart

## ⚙️ Technical Details

- **🛠️ Framework**: Powered by Streamlit
- **🧠 Model**: Pre-trained scikit-learn model (`best_model.pkl`)
- **🔧 Preprocessing**:
  - 🔠 Categorical encoding with LabelEncoder
  - 🧹 Automatic missing value handling
  - 🧩 Feature order preservation
- **✅ Input Validation**: Ensures complete feature set

## 📦 Requirements

```bash
pip install streamlit pandas scikit-learn joblib numpy
```

## 🖥️ Running the App

```bash
streamlit run app.py
```

## 🧠 Model Information

The model was trained on the Adult Census Income dataset with these features:
`[feature_1, feature_2, ...]`

## 📸 Screenshot

![App Screenshot](screenshot.png) *(optional)*

## 📜 License
This project is **open-source** under the **MIT License**. 

```

