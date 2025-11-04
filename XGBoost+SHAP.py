import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

file_path = r'path'
df = pd.read_excel(file_path, engine='openpyxl', header=0)


X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(int)


param_dist = {
    'n_estimators': [100, 200, 400, 600],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5, 1.0]
}

xgb_base = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=20, 
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("=== Performing Randomized Search CV for hyperparameter tuning... ===")
random_search.fit(X, y)
best_params = random_search.best_params_
print("\nBest Hyperparameters Found:", best_params)

# ======== 5-Fold Stratified CV with Best Hyperparameters ========
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_shap_values = []
all_test_X = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    print(f'\n=== Fold {fold} ===')
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBClassifier(
        **best_params,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    # Performance
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')

    # SHAP
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    all_shap_values.append(shap_values.values)
    all_test_X.append(X_test)


# ======== SHAP Feature Importance ========
shap_values_all = np.vstack(all_shap_values)
X_all = pd.concat(all_test_X)

mean_abs_shap = np.abs(shap_values_all).mean(axis=0)
shap_summary_df = pd.DataFrame({
    'Feature': X.columns,
    'MeanAbsSHAP': mean_abs_shap
}).sort_values(by='MeanAbsSHAP', ascending=False)

print("\nTop 10 Important Features by SHAP Value:")
print(shap_summary_df.head(10))

# Plot
shap.summary_plot(shap_values_all, X_all, feature_names=X.columns)



