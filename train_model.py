import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("customer_churn_dataset.csv")
print(df.head())
print(df.shape)
print(df.columns.tolist())
df=df.dropna()
df=df.drop_duplicates()
if "CustomerID" in df.columns:
  df=df.drop("CustomerID", axis=1)
x=df.drop("Churn", axis=1)
y=df["Churn"]
cat_cols=x.select_dtypes(
    include=["object"]
).columns.tolist()
preprocessor=ColumnTransformer(
    [
        ("cat", 
         OneHotEncoder(handle_unknown="ignore"),
         cat_cols)
    ],
    remainder="passthrough"
) 
model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
pipeline=Pipeline([
    ("preprocessor", preprocessor), ("classifier", model)
])
x_train, x_test, y_train, y_test=train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)
pipeline.fit(x_train, y_train)
prediction=pipeline.predict(x_test)
accuracy=accuracy_score(
    y_test, prediction
)
print("Model Accuracy:", accuracy)
joblib.dump(
    pipeline, "churn_model.pkl"
)
print("Model exported successfully!")