import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
DATASET_FILE = "training_dataset_baseline.csv"
MODEL_OUTPUT_FILE = "wildlife_model_rf.pkl"
PLOT_OUTPUT_FILE = "confusion_matrix.png"

if __name__ == "__main__":
    print(f"Loading dataset from {DATASET_FILE}...")
    if not os.path.exists(DATASET_FILE):
        print(f"Error: Dataset file not found at {DATASET_FILE}")
        exit()

    df = pd.read_csv(DATASET_FILE)
    df.fillna(0, inplace=True)  # Safety

    # --- Identify label column ---
    if "label" not in df.columns:
        raise ValueError("ERROR: The dataset must include a 'label' column!")

    label_col = "label"
    feature_cols = [col for col in df.columns if col != label_col]

    print(f"\nDetected {len(feature_cols)} feature columns.")

    X = df[feature_cols]
    y = df[label_col]

    print("\nClass Distribution:")
    print(y.value_counts(normalize=True))

    # --- Train-test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTraining size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # --- Model Training ---
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Wildlife", "Wildlife"]))

    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Confusion Matrix ---")
    print(cm)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Act 0", "Act 1"])
    plt.title("Confusion Matrix")
    plt.savefig(PLOT_OUTPUT_FILE)
    print(f"Confusion matrix plot saved to {PLOT_OUTPUT_FILE}")

    # --- Save Model ---
    with open(MODEL_OUTPUT_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to {MODEL_OUTPUT_FILE}")
    print("Training pipeline complete.")
