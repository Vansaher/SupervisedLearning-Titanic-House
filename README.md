# Supervised Learning Project (CSC4600)

This repository showcases an end-to-end supervised machine learning project for both **classification** and **regression** tasks using Python. It covers everything from data preprocessing and exploratory data analysis (EDA) to model training, evaluation, and visualization.

## 📁 Repository Structure

```
📦 Supervised-Learning-Project
├── 📂 Datasets             # Contains Titanic and House Price datasets
├── 📂 Visualization        # Contains visualizations from EDA and results
├── 📄 Report.pdf           # Comprehensive project report (PDF)
└── 📄 supervisedLearning.py # Python script for data analysis and modeling
```

## 📊 Overview

### Datasets

#### 1. Titanic Dataset (Classification)
- **Source**: Kaggle
- **Goal**: Predict whether a passenger survived
- **Features**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- **Target**: `Survived` (0 = Died, 1 = Survived)

#### 2. House Prices Dataset (Regression)
- **Goal**: Predict the price of a house
- **Features**: Rooms, Distance
- **Target**: `Value` (Price)

---

## 🛠️ Methods Used

- Data cleaning and preprocessing (e.g. missing value handling, encoding)
- Exploratory Data Analysis (EDA)
- Supervised learning models:
  - Logistic Regression & Random Forest Classifier (Titanic)
  - Linear Regression & Random Forest Regressor (House Prices)
- Model evaluation:
  - **F1 Score** (for classification)
  - **RMSE & R² Score** (for regression)
- Visualization with Matplotlib and Seaborn

---

## 📈 Results Summary

### Titanic Dataset (Classification)
| Model                | F1 Score |
|---------------------|----------|
| Logistic Regression | 0.764    |
| Random Forest       | 0.775    |

🎯 **Best Model**: Random Forest Classifier

---

### House Price Dataset (Regression)
| Model              | RMSE  | R² Score |
|-------------------|-------|----------|
| Linear Regression | 3.994 | 0.768    |
| Random Forest     | 3.383 | 0.833    |

🎯 **Best Model**: Random Forest

---


## 📌 Key Learnings

- Importance of proper preprocessing in improving model outcomes
- Power of ensemble methods (Random Forest) in capturing non-linear relationships
- Visual insights can guide more informed modeling decisions

---

## 👨‍🎓 Author

**Mohammad Javan Samboeputra Herlambang**  
Faculty of Computer Science and Information Technology, UPM  

---

This project is for educational purposes.
