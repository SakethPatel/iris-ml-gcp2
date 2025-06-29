import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

data = pd.read_csv("dataset/iris.csv")
X = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "artifacts/model.joblib")
