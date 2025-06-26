import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

# Load images and labels
def load_data(path, size=(48, 48)):  # reduced size for speed
    X, y = [], []
    for label, cls in enumerate(['benign', 'malignant']):
        folder = os.path.join(path, cls)
        for file in tqdm(os.listdir(folder), desc=f"Loading {cls}"):
            img = imread(os.path.join(folder, file))
            img = resize(img, size).flatten()
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

# Load train data
X, y = load_data("C:/Users/reshm/OneDrive/Desktop/dataset/data/train")


# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Models (optimized for speed)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_jobs=-1),
    "Linear SVM": LinearSVC(max_iter=1000),
    "KNN": KNeighborsClassifier(n_jobs=-1),
    "Naive Bayes": GaussianNB()
}

results = []
best_f1 = 0
best_model = None
best_model_name = ""

# Train models
for name, model in models.items():
    print(f"🔄 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    results.append((name, acc, f1, prec))
    print(f"✅ {name} Done - Acc: {acc:.2f}, F1: {f1:.2f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# Save best model
joblib.dump(best_model, "bestmodel.pkl")

# Save results
import pandas as pd
df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score", "Precision"])
df.to_csv("model_metrics.csv", index=False)

print(f"🏆 Best model: {best_model_name} (F1: {best_f1:.2f}) saved to bestmodel.pkl")

# Load train data

