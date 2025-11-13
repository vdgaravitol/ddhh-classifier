import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import datetime

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------
# Obtiene la ruta absoluta del modelo
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "models", "BeRT_Training"))



# Mapeo de etiquetas al esquema de semáforo
LABEL_MAP = {
    2: "🟢 VERDE",
    1: "🟡 AMARILLO",
    0: "🔴 ROJO"
}

# ------------------------------------------------------------
# FUNCIÓN: CARGAR MODELO Y TOKENIZADOR
# ------------------------------------------------------------
def load_model(model_dir=MODEL_DIR):
    print(f"Cargando modelo desde: {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

# ------------------------------------------------------------
# FUNCIÓN: PREPROCESAMIENTO BÁSICO DE TEXTO
# ------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\n", " ")
    return text

# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE CLASIFICACIÓN
# ------------------------------------------------------------
def classify_text(text: str, tokenizer, model):
    """
    Clasifica un texto según el modelo semáforo DDHH.
    Retorna la etiqueta, la confianza y un timestamp.
    """
    clean_text = preprocess_text(text)

    # Tokenizar el texto
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Inferencia sin gradientes
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    result = {
        "texto": text,
        "clasificacion": LABEL_MAP[pred_label],
        "confianza": round(confidence, 4),
        "timestamp": datetime.datetime.now().isoformat()
    }

    return result

# ------------------------------------------------------------
# BLOQUE PRINCIPAL (EJECUCIÓN LOCAL)
# ------------------------------------------------------------
if __name__ == "__main__":
    tokenizer, model = load_model()

    print("\n Clasificador de Tweets para Protección de DDHH")
    print("Escribe un texto o tweet para analizar (máx 300 caracteres).")
    print("Escribe 'salir' para terminar.\n")

    while True:
        text = input("> ")

        if text.lower().strip() in ["salir", "exit", "quit"]:
            print("Saliendo del clasificador...")
            break

        if not text.strip():
            print("Por favor ingresa un texto válido.")
            continue

        result = classify_text(text, tokenizer, model)

        print("\n--- RESULTADO ---")
        print(f"Texto: {result['texto']}")
        print(f"Clasificación: {result['clasificacion']}")
        print(f"Confianza: {result['confianza']}")
        print(f"Timestamp: {result['timestamp']}\n")
