#Maiyongyi-234-10-M3_Exercise3
# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===================== Part 1: Load and inspect the dataset =====================
print("===== 1. Load Dataset =====")
df = sns.load_dataset("titanic")

print("\n===== 2. First 5 rows =====")
print(df.head())

print("\n===== 3. Dataset shape (rows, columns) =====")
print(df.shape)

print("\n===== 4. Column names =====")
print(df.columns.tolist())

print("\n===== 5. Data types =====")
print(df.dtypes)

print("\n===== 6. Missing values count =====")
print(df.isnull().sum())

# ===================== Part 2: Feature Selection =====================
print("\n===== 7. Select useful features =====")

# 只保留 纯字符串/纯数值 特征，彻底避开 category 类型报错！
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]

# Drop rows with missing values for simplicity (SAFEST FIX)
df = df.dropna()

# Define features X and target y
X = df.drop("survived", axis=1)
y = df["survived"]

# Separate numerical and categorical features
numeric_features = ["age", "sibsp", "parch", "fare", "pclass"]
categorical_features = ["sex", "embarked"]

print("Shape of features X:", X.shape)
print("Shape of target y:", y.shape)

# ===================== Part 3: Preprocessing Pipeline =====================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ===================== Part 4: Model =====================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# ===================== Part 5: Evaluation =====================
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("===== Model Evaluation Results =====")
print("="*50)

print("\n1. Accuracy:")
print(round(accuracy_score(y_test, y_pred), 4))

print("\n2. Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n3. Classification Report:")
print(classification_report(y_test, y_pred))
