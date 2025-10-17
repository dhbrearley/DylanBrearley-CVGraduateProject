import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

DATA_PATH = "../Data/metrics_labeled_updated.csv"
MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model_noACS.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_noACS.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df_trainable = df[df['label'].isin(['good', 'bad'])].copy()

# keep out acs to i have data to test on
mask_non_acs = ~df_trainable['video_id'].str.contains("ACS", case=False, na=False)
df_trainable = df_trainable[mask_non_acs]

print(f"Training set excludes ACS videos.")
print(f"Remaining videos: {df_trainable['video_id'].nunique()}")
print(f"Remaining samples: {len(df_trainable)}")

feature_cols = ["sharpness", "glare", "contrast", "entropy", 
                "edge_density", "mean_brightness", "saturation"]

scaler = MinMaxScaler()
df_trainable[feature_cols] = scaler.fit_transform(df_trainable[feature_cols])

X = df_trainable[feature_cols]
y = df_trainable['label'].map({"bad": 0, "good": 1})

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
sample_weights = np.array([class_weights[label] for label in y])

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

print("\nTraining model...")
clf.fit(X, y, sample_weight=sample_weights)
print("Training done.")

clf.save_model(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
