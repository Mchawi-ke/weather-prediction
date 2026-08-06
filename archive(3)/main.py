import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Loading dataset
df = pd.read_csv("seattle_weather.csv")
print(df.head(5))

x = df[["precipitation", "temp_max", "temp_min", "wind"]]
y = df["weather"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

gb.fit(x_train, y_train)

y_pred_gb = gb.predict(x_test)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb, target_names=le.classes_))


print(df.head())


new_day = pd.DataFrame([{
    "precipitation": 0.0,
    "temp_max": 16.5,
    "temp_min": 6.0,
    "wind": 3.2
}])

pred = rf.predict(new_day)
print("Tahmin:", le.inverse_transform(pred))



