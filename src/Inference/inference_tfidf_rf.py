"""
inference_tfidf_rf.py
----------------------------------
Realiza inferencia con el modelo tradicional TF-IDF + Random Forest
para clasificar texto en tres categorías (Hate Speech, Offensive Language, Neither)
y mostrar un resultado tipo semáforo.

Autor: Vivian Garavito
"""

# =====================================================
# === Librerías ===
# =====================================================
import os
import joblib
import re
import sys
import datetime

# =====================================================
# === Configuración (rutas dinámicas y seguras) ===
# =====================================================

# Directorio base absoluto del proyecto (sube desde /src/Inference)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "TF-IDF", "random_forest_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "data", "models", "TF-IDF", "tfidf_vectorizer.pkl")

# =====================================================
# === Verificación de existencia de archivos ===
# =====================================================
if not os.path.exists(MODEL_PATH):
    print(f" No se encontró el modelo en: {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(VECTORIZER_PATH):
    print(f" No se encontró el vectorizador en: {VECTORIZER_PATH}")
    sys.exit(1)

# Cargar modelo y vectorizador
print("✅ Cargando modelo y vectorizador TF-IDF + Random Forest...")
rf_model = joblib.load(MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
print("Modelo cargado correctamente.\n")

# =====================================================
# === Mapeo de etiquetas a niveles de semáforo ===
# =====================================================
label_map = {
    "Hate Speech": ("🔴", "Hate Speech"),
    "Offensive Language": ("🟡", "Offensive Language"),
    "Neither": ("🟢", "Neither")
}

# =====================================================
# === Función de limpieza básica ===
# =====================================================
def clean_text(text: str) -> str:
    """Limpieza ligera para normalizar el texto."""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"#\w+", "<HASHTAG>", text)
    text = re.sub(r"[^a-z\s<>]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# =====================================================
# === Función principal de inferencia ===
# =====================================================
def classify_text_rf(text: str):
    """
    Clasifica un texto usando el modelo TF-IDF + Random Forest
    y devuelve la etiqueta tipo semáforo.
    """
    clean = clean_text(text)
    text_tfidf = tfidf_vectorizer.transform([clean])

    prediction = rf_model.predict(text_tfidf)[0]
    confidence = float(rf_model.predict_proba(text_tfidf).max())

    emoji, label_desc = label_map.get(prediction, ("⚪", "Unknown"))

    return {
        "input_text": text,
        "label": label_desc,
        "semaforo": emoji,
        "confidence": round(confidence, 3),
        "timestamp": datetime.datetime.now().isoformat()
    }

# =====================================================
# === Ejecución interactiva ===
# =====================================================
if __name__ == "__main__":
    print(" Clasificador TF-IDF + Random Forest")
    print("Escribe un texto en inglés para clasificar (o 'salir' para terminar):\n")

    while True:
        user_input = input("Tweet o texto: ").strip()
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Saliendo del clasificador...")
            break

        if not user_input:
            print("Texto vacío, intenta de nuevo.\n")
            continue

        result = classify_text_rf(user_input)

        print("\n--- RESULTADO ---")
        print(f"Texto: {result['input_text']}")
        print(f"Clasificación: {result['semaforo']} {result['label']}")
        print(f"Confianza: {result['confidence']}")
        print(f"Timestamp: {result['timestamp']}\n")
