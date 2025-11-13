"""
model_training_TF-IDF_RF.py
----------------------------------
Entrena y compara modelos tradicionales (Logistic Regression, SVM, Random Forest)
usando representaciones TF-IDF, y guarda únicamente el modelo Random Forest entrenado.

Entradas:
    - data/processed/balanced_data.parquet

Salidas:
    - Modelo Random Forest guardado en data/models/TF-IDF_RF/
    - Métricas comparativas guardadas en data/traditional_models_compare.csv
"""

# =====================================================
# === Librerías Base ===
# =====================================================
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =====================================================
# === Configuración Inicial ===
# =====================================================
DATA_PATH = "../data/processed/balanced_data.parquet"
MODEL_PATH = "../data/models/TF-IDF/"

METRICS_PATH = "../data/traditional_models_compare.csv"

os.makedirs(MODEL_PATH, exist_ok=True)

# =====================================================
# === Carga del Dataset ===
# =====================================================
data = pd.read_parquet(DATA_PATH)
print(f"Dataset cargado correctamente ({data.shape[0]} registros).")

X = data["tweet"]
y = data["label_desc"]

# División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Split -> Train: {len(X_train)}, Test: {len(X_test)}")

# =====================================================
# === Vectorización TF-IDF ===
# =====================================================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
print(f"Vectorización completada ({X_train_vec.shape[1]} features).")

# =====================================================
# === Entrenamiento y Evaluación de Modelos ===
# =====================================================
results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_vec, y_train)
y_pred_lr = lr.predict(X_test_vec)
results["Logistic Regression"] = classification_report(y_test, y_pred_lr, output_dict=True)

# Support Vector Machine
svm = LinearSVC(random_state=42)
svm.fit(X_train_vec, y_train)
y_pred_svm = svm.predict(X_test_vec)
results["SVM"] = classification_report(y_test, y_pred_svm, output_dict=True)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_vec, y_train)
y_pred_rf = rf.predict(X_test_vec)
results["Random Forest"] = classification_report(y_test, y_pred_rf, output_dict=True)

print("\nEntrenamiento de modelos completado.")

# =====================================================
# === Consolidación de Métricas ===
# =====================================================
summary_df = pd.DataFrame({
    model: {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"]
    }
    for model, report, pred in [
        ("Logistic Regression", results["Logistic Regression"], y_pred_lr),
        ("SVM", results["SVM"], y_pred_svm),
        ("Random Forest", results["Random Forest"], y_pred_rf)
    ]
}).T.round(3)

print("\nResultados comparativos de desempeño:")
print(summary_df)

# Guardar métricas en CSV dentro de /data/
summary_df.to_csv(METRICS_PATH)
print(f"\n Métricas guardadas en: {METRICS_PATH}")

# =====================================================
# === Guardar solo el modelo Random Forest ===
# =====================================================
joblib.dump(rf, os.path.join(MODEL_PATH, "random_forest_model.pkl"))
joblib.dump(tfidf, os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
print(f"Modelo Random Forest y vectorizador guardados en: {MODEL_PATH}")

print("\n Script finalizado correctamente.")
