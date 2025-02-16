# Imports and data loading

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import os
os.environ['PYTHONHASHSEED'] = '42'

warnings.simplefilter(action = 'ignore', category = FutureWarning)

random.seed(24)
np.random.seed(24)
tf.random.set_seed(24)

# Load data

df = pd.read_csv("heart_disease_uci.csv")
print("Initial df shape:", df.shape)

# Data Inspection
print("DataFrame Info:")
df.info()
print("\nData Types:\n", df.dtypes)
print(df.head())

# Convert columns to categorical/boolean
categorical_cols = ["sex", "cp", "restecg", "slope", "thal"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

bool_cols = ["fbs", "exang"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype("bool")

df.info()

# If column "ca" misses more than 500 rows, then drop it
if 'ca' in df.columns:
    missing_ca = df['ca'].isnull().sum()
    if missing_ca > 500:
        df.drop('ca', axis=1, inplace=True)

# Impute other numeric columns with median
num_cols = df.select_dtypes(include=['float64','int64']).columns
print("Numeric columns for imputation:", list(num_cols))

imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Add "missing" category and fill for categorical columns
cat_cols = df.select_dtypes(include=['category']).columns
print("Categorical columns to fill with 'missing':", list(cat_cols))

# Add a new 'missing' category
for col in cat_cols:
    df[col] = df[col].cat.add_categories(["missing"])
    df[col] = df[col].fillna("missing")

# Fill missing with False for all the boolean columns
bool_cols = df.select_dtypes(include=['bool']).columns
print("Boolean columns to fill with False:", list(bool_cols))
df[bool_cols] = df[bool_cols].fillna(False)

# Confirm that there aren't any columns with missing values left
missing_counts = df.isnull().sum()
print("After Full Imputation & 'missing' Category:")
print(missing_counts)
print("Remaining rows:", len(df))

cols_to_drop = ["id"]
df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

# Create missing categories and binarize target
df["num"] = (df["num"] >= 1).astype(int)
print("\nTarget distribution:\n", df["num"].value_counts())

# One-Hot Encoding
categorical_feats = ["sex", "cp", "restecg", "slope", "thal", "dataset"]
cat_to_encode = [col for col in categorical_feats if col in df.columns]
df_encoded = pd.get_dummies(df, columns=cat_to_encode, drop_first=True)
print(df_encoded.shape)
print(df_encoded.columns)

# Split into X and y

y = df_encoded["num"]
X = df_encoded.drop("num", axis=1)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)

print("\nSplit sizes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# Scale numeric columns
numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
scaler = StandardScaler()
X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])


# Logistic regression analysis

logreg = LogisticRegression(max_iter=200, random_state = 24)
logreg.fit(X_train, y_train)
lr_preds = logreg.predict(X_test)

print("LR Accuracy on test:", logreg.score(X_test, y_test))
print("Classification Report (LR):\n", classification_report(y_test, lr_preds))
print("Confusion Matrix (LR):\n", confusion_matrix(y_test, lr_preds))

# MLP #1 (Dropout=0.2)

model1 = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model1.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history1 = model1.fit(
    X_train, y_train,
    epochs=50, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

test_loss1, test_acc1 = model1.evaluate(X_test, y_test)
print("MLP #1 Test Accuracy:", test_acc1)

mlp1_probs = model1.predict(X_test)
mlp1_preds = (mlp1_probs > 0.5).astype(int)

print("Classification Report (MLP #1):\n", classification_report(y_test, mlp1_preds))
print("Confusion Matrix (MLP #1):\n", confusion_matrix(y_test, mlp1_preds))

# MLP #2 (Dropout=0.5)

model2 = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model2.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

early_stop2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history2 = model2.fit(
    X_train, y_train,
    epochs=50, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop2],
    verbose=1
)

test_loss2, test_acc2 = model2.evaluate(X_test, y_test)
print("MLP #2 (Dropout=0.5) Test Accuracy:", test_acc2)

mlp2_probs = model2.predict(X_test)
mlp2_preds = (mlp2_probs > 0.5).astype(int)

print("Classification Report (MLP #2):\n", classification_report(y_test, mlp2_preds))
print("Confusion Matrix (MLP #2):\n", confusion_matrix(y_test, mlp2_preds))

# Convergence function

def summarize_convergence(history, model_name="MLP"):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)
    train_loss = hist['loss']
    val_loss = hist.get('val_loss', None)
    train_acc = hist.get('accuracy', None)
    val_acc = hist.get('val_accuracy', None)
    
    data = {'epoch': list(epochs), 'train_loss': train_loss}
    if val_loss is not None:
        data['val_loss'] = val_loss
    if train_acc is not None:
        data['train_accuracy'] = train_acc
    if val_acc is not None:
        data['val_accuracy'] = val_acc
    
    df_hist = pd.DataFrame(data)
    
    print(f"\n=== Convergence Data for {model_name} ===\n")
    print(df_hist.to_string(index=False))
    
    # Plot loss function
    plt.figure(figsize=(8,4))
    plt.plot(epochs, train_loss, label='Train Loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss, label='Val Loss')
    plt.title(f"{model_name}: Loss vs. Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Plot accuracy
    if train_acc is not None:
        plt.figure(figsize=(8,4))
        plt.plot(epochs, train_acc, label='Train Accuracy')
        if val_acc is not None:
            plt.plot(epochs, val_acc, label='Val Accuracy')
        plt.title(f"{model_name}: Accuracy vs. Epoch")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

summarize_convergence(history1, "MLP #1 (Dropout=0.2)")
summarize_convergence(history2, "MLP #2 (Dropout=0.5)")

# Confusion matrix heatmaps

def plot_confusion_matrix(cm, model_name="Model"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Logistic regression heatmap
cm_lr = confusion_matrix(y_test, lr_preds)
plot_confusion_matrix(cm_lr, "Logistic Regression")

# MLP #1 heatmap
cm_mlp1 = confusion_matrix(y_test, mlp1_preds)
plot_confusion_matrix(cm_mlp1, "MLP #1 (Dropout=0.2)")

# MLP #2 heatmap
cm_mlp2 = confusion_matrix(y_test, mlp2_preds)
plot_confusion_matrix(cm_mlp2, "MLP #2 (Dropout=0.5)")

# SHAP Analysis for MLP #1
try:
    X_train_float = X_train.astype('float32')
    background_kernel = X_train_float.sample(50, random_state=42)
    def model1_predict(data):
        return model1.predict(data).ravel()
    explainer_kernel = shap.KernelExplainer(
        model1_predict, 
        background_kernel, 
        link="logit"
    )

    X_test_sample = X_test.sample(10, random_state=42).astype('float32')
    shap_values_kernel = explainer_kernel.shap_values(X_test_sample, nsamples="auto")
    if isinstance(shap_values_kernel, list):
        shap_values_kernel = shap_values_kernel[0]
    
    df_shap = pd.DataFrame(shap_values_kernel, columns=X_test_sample.columns)
    print("SHAP values DataFrame:\n", df_shap)
    
    preds_kernel = model1_predict(X_test_sample)
    df_shap["pred_prob"] = preds_kernel
    print("\nSHAP values + predicted probabilities:\n", df_shap)
    
    shap.summary_plot(
        shap_values_kernel, 
        X_test_sample, 
        feature_names=X_test_sample.columns.tolist()
    )
    
except Exception as e:
    print("KernelExplainer SHAP failed:", e)

# Logistic regression metrics
lr_report_dict = classification_report(y_test, lr_preds, output_dict=True)
lr_accuracy = logreg.score(X_test, y_test)
lr_precision = lr_report_dict['macro avg']['precision']
lr_recall    = lr_report_dict['macro avg']['recall']
lr_f1        = lr_report_dict['macro avg']['f1-score']

# 2) MLP #1 metrics
mlp1_report_dict = classification_report(y_test, mlp1_preds, output_dict=True)
mlp1_accuracy = test_acc1
mlp1_precision = mlp1_report_dict['macro avg']['precision']
mlp1_recall    = mlp1_report_dict['macro avg']['recall']
mlp1_f1        = mlp1_report_dict['macro avg']['f1-score']

# 3) MLP #2 metrics
mlp2_report_dict = classification_report(y_test, mlp2_preds, output_dict=True)
mlp2_accuracy = test_acc2
mlp2_precision = mlp2_report_dict['macro avg']['precision']
mlp2_recall    = mlp2_report_dict['macro avg']['recall']
mlp2_f1        = mlp2_report_dict['macro avg']['f1-score']

summary_data = {
    'Model': ['Logistic Regression', 'MLP #1 (dropout=0.2)', 'MLP #2 (dropout=0.5)'],
    'Accuracy': [lr_accuracy, mlp1_accuracy, mlp2_accuracy],
    'Precision (macro)': [lr_precision, mlp1_precision, mlp2_precision],
    'Recall (macro)': [lr_recall, mlp1_recall, mlp2_recall],
    'F1 (macro)': [lr_f1, mlp1_f1, mlp2_f1]
}

df_summary = pd.DataFrame(summary_data)
print("\nFinal Summary Table:\n")
print(df_summary.to_string(index=False))
df_summary
