import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

titanic = pd.read_csv("titanic.csv")
house = pd.read_csv("house_prices.csv")

# ========== Preprocessing: TITANIC ==========
titanic = titanic.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

le = LabelEncoder()
titanic['Sex'] = le.fit_transform(titanic['Sex'])  # male=1, female=0
titanic['Embarked'] = le.fit_transform(titanic['Embarked'])

# ========== TERMINAL-BASED EDA: TITANIC ==========
print("\n===== TITANIC DATASET EDA =====")

print("\n[1] Survival Count:")
print(titanic['Survived'].value_counts())

print("\n[2] Survival Rate by Sex:")
print(titanic.groupby('Sex')['Survived'].mean())

print("\n[3] Survival Rate by Pclass:")
print(titanic.groupby('Pclass')['Survived'].mean())

print("\n[4] Statistical Summary:")
print(titanic.describe())

print("\n[5] Correlation with Survival:")
print(titanic.corr(numeric_only=True)['Survived'].sort_values(ascending=False))

# ========== PLOTS: TITANIC ==========
sns.countplot(x='Survived', hue='Sex', data=titanic)
plt.title('Survival by Sex')
plt.show()

sns.histplot(titanic['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm')
plt.title('Titanic Correlation Heatmap')
plt.show()

# ========== MODELING: TITANIC ==========
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_log = logreg.predict(X_test)
y_pred_rf = rf.predict(X_test)

print("\nLogistic Regression F1:", f1_score(y_test, y_pred_log))
print("Random Forest F1:", f1_score(y_test, y_pred_rf))


# ========== Preprocessing: HOUSE ==========
X_house = house[['Rooms', 'Distance']]
y_house = house['Value']

scaler = StandardScaler()
X_house_scaled = scaler.fit_transform(X_house)

X_train, X_test, y_train, y_test = train_test_split(X_house_scaled, y_house, test_size=0.2, random_state=42)

# ========== TERMINAL-BASED EDA: HOUSE PRICE ==========
print("\n===== HOUSE PRICE DATASET EDA =====")

print("\n[1] Statistical Summary:")
print(house.describe())

print("\n[2] Correlation with Value:")
print(house.corr()['Value'].sort_values(ascending=False))

print("\n[3] Distribution of Value (Grouped):")
print(pd.cut(house['Value'], bins=5).value_counts())

# ========== PLOTS: HOUSE ==========
sns.pairplot(house)
plt.show()

sns.heatmap(house.corr(), annot=True, cmap='coolwarm')
plt.title('House Price Correlation Heatmap')
plt.show()

# ========== MODELING: HOUSE PRICE ==========
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()
rf_reg = RandomForestRegressor(random_state=42)

lr.fit(X_train, y_train)
rf_reg.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)

print("\nLinear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

print("\nLinear Regression R2:", r2_score(y_test, y_pred_lr))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
