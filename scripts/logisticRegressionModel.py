import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler


train_df = pd.read_csv("train_bbbp.csv")
valid_df = pd.read_csv("valid_bbbp.csv")

##ensures training is only on features no matter what they're called.
non_feature_cols = ['ids', 'y', 'w']
feature_cols = [col for col in train_df.columns if col not in non_feature_cols]

X_train = train_df[feature_cols].values
y_train = train_df["y"].values
X_valid = valid_df[feature_cols].values
y_valid = valid_df["y"].values

# scale features
# It's important to scale features for logistic regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')  # You could also try 'saga' for larger datasets
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_valid)
probs = model.predict_proba(X_valid)[:, 1]

# Evaluation
print(f"Accuracy: {accuracy_score(y_valid, preds):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_valid, probs):.3f}")
print("\nDetailed Classification Report:")
print(classification_report(y_valid, preds))


# testing starts here
test_df = pd.read_csv("test_bbbp.csv")
X_test = test_df[feature_cols].values
X_test = scaler.transform(X_test)  

# Predict
test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]
output_df = pd.DataFrame({
    "ids": test_df["ids"],
    "predicted_label": test_preds,
    "predicted_prob": test_probs
})

output_df.to_csv("test_predictions.csv", index=False)
