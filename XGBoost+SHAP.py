import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

file_path = r'path'
df = pd.read_excel(file_path, engine='openpyxl', header=0)


X = df.iloc[]
y = df.iloc[].astype(int)  # （0 or 1）


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_shap_values = []
all_test_X = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    print(f'\n=== Fold {fold} ===')

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    all_shap_values.append(shap_values.values)
    all_test_X.append(X_test)

shap_values_all = np.vstack(all_shap_values)
X_all = pd.concat(all_test_X)


mean_abs_shap = np.abs(shap_values_all).mean(axis=0)
shap_summary_df = pd.DataFrame({
    'Feature': X.columns,
    'MeanAbsSHAP': mean_abs_shap
}).sort_values(by='MeanAbsSHAP', ascending=False)

print("\nTop 8 important features by SHAP value:")
print(shap_summary_df.head(10))


shap.summary_plot(shap_values_all, X_all, feature_names=X.columns)

