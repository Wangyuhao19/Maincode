import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import uniform, randint

file_path = r'path'
df = pd.read_excel(file_path, engine='openpyxl', header=0)

X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(int)


param_dist = {
    "n_estimators": list(range(50, 301, 10)),  # 50, 60, ..., 300
    "learning_rate": uniform(0.01, 0.29),      # [0.01, 0.30]
    "max_depth": randint(2, 11)                # {2,3,...,10}
}

# Base model: keep other params as default
xgb_base = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)


cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=200,          # <-- required
    scoring="f1",        # recommended; change to "accuracy" if needed
    cv=cv5,
    verbose=1,
    n_jobs=-1,
    random_state=42,
    return_train_score=True
)

print("=== Performing RandomizedSearchCV for hyperparameter tuning... ===")
random_search.fit(X, y)

best_params = random_search.best_params_
best_score = random_search.best_score_
print("\nBest Hyperparameters Found:", best_params)
print(f"Best mean CV score ({random_search.scoring}): {best_score:.4f}")


all_shap_values = []
all_test_X = []

for fold, (train_idx, test_idx) in enumerate(cv5.split(X, y), 1):
    print(f"\n=== Fold {fold} ===")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Performance (report both)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # F1 averaging: binary vs multiclass
    unique_classes = np.unique(y)
    if len(unique_classes) == 2:
        f1 = f1_score(y_test, y_pred, average="binary")
    else:
        f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # SHAP
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    all_shap_values.append(shap_values.values)
    all_test_X.append(X_test)


shap_values_all = np.vstack(all_shap_values)
X_all = pd.concat(all_test_X, axis=0)

mean_abs_shap = np.abs(shap_values_all).mean(axis=0)
shap_summary_df = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsSHAP": mean_abs_shap
}).sort_values(by="MeanAbsSHAP", ascending=False)

print("\nTop 10 Important Features by Mean(|SHAP|):")
print(shap_summary_df.head(10))


shap.summary_plot(shap_values_all, X_all, feature_names=X.columns)
plt.show()



