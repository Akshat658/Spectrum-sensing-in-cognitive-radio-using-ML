# -*- coding: utf-8 -*
"""
SVM training for single-channel spectrum sensing
-----------------------------------------------
Dataset: spectrum_dataset_single.csv

Columns (from MATLAB):
0: Energy (normalized)
1: SNR_dB
2: Label (0 = PU absent, 1 = PU present)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib   # pip install joblib if needed


# ============================================================
# 1. Load dataset
# ============================================================
print("Loading dataset...")

data = pd.read_csv("Path_to_dataset", header=None)
data.columns = ["energy", "snr_db", "label"]

print(data.head())
print("Total samples:", len(data))

# Features and labels
X = data[["energy", "snr_db"]].values   # shape: (N, 2)
y = data["label"].values               # shape: (N,)


# ============================================================
# 2. Train / test split
# ============================================================
print("\nSplitting into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))


# ============================================================
# 3. Feature scaling
# ============================================================
print("\nScaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ============================================================
# 4. Train SVM (RBF kernel)
# ============================================================
print("\nTraining SVM (RBF kernel) model...")

model = SVC(
    kernel="rbf",
    C=10.0,         # can tune later
    gamma="scale",
    probability=False
)

model.fit(X_train_scaled, y_train)

print("Training complete.")


# ============================================================
# 5. Evaluation on test set
# ============================================================
print("\nEvaluating model on test set...")

y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print(f"\nTest Accuracy (SVM, single-channel): {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))


# ============================================================
# 6. SNR vs Accuracy (since you used multiple SNRs in MATLAB)
# ============================================================
print("\nComputing accuracy vs SNR...")

snr_values = sorted(data["snr_db"].unique())
snr_accuracies = []

for snr in snr_values:
    mask = data["snr_db"] == snr
    X_snr = data.loc[mask, ["energy", "snr_db"]].values
    y_snr = data.loc[mask, "label"].values

    X_snr_scaled = scaler.transform(X_snr)
    y_snr_pred   = model.predict(X_snr_scaled)

    snr_acc = accuracy_score(y_snr, y_snr_pred)
    snr_accuracies.append(snr_acc)

plt.figure()
plt.plot(snr_values, snr_accuracies, marker="o")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.title("SNR vs Detection Accuracy (SVM, single-channel)")
plt.grid(True)
plt.tight_layout()
plt.savefig("snr_vs_accuracy_svm_single.png", dpi=300)
plt.show()

print("Saved SNR vs accuracy plot as snr_vs_accuracy_svm_single.png")


# ============================================================
# 7. Save model + scaler for deployment
# ============================================================
print("\nSaving model and scaler...")

joblib.dump(model, "spectrum_model_svm_single.pkl")
joblib.dump(scaler, "spectrum_scaler_svm_single.pkl")

print("Saved model to  : spectrum_model_svm_single.pkl")
print("Saved scaler to : spectrum_scaler_svm_single.pkl")
