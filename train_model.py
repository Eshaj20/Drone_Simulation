import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load data
df = pd.read_csv("drone_activity_dataset.csv")

# Encode labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# Feature Engineering
df["speed_altitude_ratio"] = df["speed"] / df["altitude"].replace(0, 1e-6)
df["proximity_score"] = 1 / df["distance_to_restricted"].replace(0, 1e-6)
df["altitude_proximity_ratio"] = df["altitude"] / df["distance_to_restricted"].replace(0, 1e-6)
df["speed_squared"] = df["speed"] ** 2
df["altitude_squared"] = df["altitude"] ** 2

# Features used
features = ["lat", "lon", "altitude", "speed", "distance_to_restricted",
            "speed_altitude_ratio", "proximity_score", "altitude_proximity_ratio",
            "speed_squared", "altitude_squared"]

X = df[features]
y = df["label_encoded"]

# Apply SMOTE to balance
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=5, learning_rate=0.1)
model.fit(X_train_scaled, y_train)
joblib.dump(model, "models/xgb_drone_classifier.pkl")

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


