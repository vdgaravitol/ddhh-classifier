"""
data_processing.py
-------------------
Script para el preprocesamiento y balanceo de datos del proyecto:
"Clasificador de Texto para Protección de Derechos Humanos"

Este módulo:
1. Carga el dataset original.
2. Limpia el texto (URLs, menciones, signos, stopwords, etc.).
3. Realiza una ampliación (data augmentation) para manejar el desbalance.
4. Genera un dataset final balanceado y lo guarda en data/processed/.

Autor: Vivian Garavito
"""

# === Librerías base ===
import pandas as pd                 # manejo de datos
import numpy as np                  # operaciones numéricas

# === Visualización ===
import matplotlib.pyplot as plt     # gráficos básicos

# === Procesamiento de texto ===
import re                           # limpieza con expresiones regulares
from nltk.corpus import stopwords, wordnet   # stopwords y sinónimos (WordNet)
from nltk import download            # descarga de recursos NLTK

# === Utilidades ===
import random                        # control de aleatoriedad
import os                       # manejo de rutas y sistema de archivos
# =====================================================
# CONFIGURACIONES INICIALES
# =====================================================
download('stopwords')
download('wordnet')
stop_words = set(stopwords.words('english'))

RAW_DATA_PATH = "../data/raw/labeled_data.csv"
PROCESSED_PATH = "../data/processed/balanced_data.csv"

# =====================================================
# FUNCIONES DE LIMPIEZA DE TEXTO
# =====================================================
def clean_text(text: str) -> str:
    """
    Limpia texto eliminando URLs, menciones, caracteres especiales, 
    convierte a minúsculas y remueve stopwords.
    """
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # URLs
    text = re.sub(r'@\w+', '', text)                    # menciones
    text = re.sub(r'[^\w\s]', '', text)                 # signos y números
    text = text.lower().strip()
    tokens = re.findall(r'\b\w+\b', text)
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(filtered_tokens)

# =====================================================
# FUNCIONES DE DATA AUGMENTATION
# =====================================================
def get_synonyms(word):
    """Obtiene sinónimos simples usando WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def augment_data(text, num_augmentations=1):
    """
    Genera versiones alternativas del texto reemplazando palabras por sinónimos.
    """
    words = text.split()
    augmented_texts = []

    for _ in range(num_augmentations):
        if not words:
            continue
        new_words = words.copy()
        idx = random.randint(0, len(new_words) - 1)
        synonyms = get_synonyms(new_words[idx])
        if synonyms:
            new_words[idx] = random.choice(synonyms)
        augmented_texts.append(" ".join(new_words))
    return augmented_texts

# =====================================================
# FUNCIONES DE BALANCEO DE CLASES
# =====================================================
def augment_to_target(df_class, target_count):
    """
    Genera nuevas muestras por clase hasta alcanzar el número objetivo.
    """
    if len(df_class) >= target_count:
        return df_class.sample(n=target_count, random_state=42)

    samples_needed = target_count - len(df_class)
    samples_to_augment = df_class.sample(n=samples_needed, replace=True, random_state=42)
    augmented = []
    for _, row in samples_to_augment.iterrows():
        new_text = augment_data(row['tweet'], num_augmentations=1)[0]
        augmented.append({
            'tweet': new_text,
            'label': row['label'],
            'label_desc': row['label_desc']
        })
    return pd.concat([df_class, pd.DataFrame(augmented)], ignore_index=True)

# =====================================================
# PIPELINE PRINCIPAL
# =====================================================
def process_data():
    """Ejecuta todo el flujo de preprocesamiento y guardado."""
    print("🔹 Cargando dataset original...")
    data = pd.read_csv(RAW_DATA_PATH)
    data = data.drop(columns=['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'])
    data.rename(columns={'class': 'label'}, inplace=True)
    label_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    data['label_desc'] = data['label'].map(label_mapping)

    print("🔹 Limpieza de texto...")
    data['tweet'] = data['tweet'].apply(clean_text)

    print("🔹 Augmentando clase minoritaria (Hate Speech)...")
    hate_speech = data[data['label'] == 0]
    augmented_samples = []
    for _, row in hate_speech.iterrows():
        for aug_text in augment_data(row['tweet'], 3):
            augmented_samples.append({'tweet': aug_text, 'label': 0, 'label_desc': 'Hate Speech'})
    data_aug = pd.concat([data, pd.DataFrame(augmented_samples)], ignore_index=True)

    print("🔹 Balanceando clases...")
    df_hate = data_aug[data_aug['label'] == 0]
    df_off = data_aug[data_aug['label'] == 1]
    df_nei = data_aug[data_aug['label'] == 2]
    target_count = len(df_off)
    df_hate_bal = augment_to_target(df_hate, target_count)
    df_nei_bal = augment_to_target(df_nei, target_count)
    data_balanced = pd.concat([df_off, df_hate_bal, df_nei_bal], ignore_index=True).sample(frac=1, random_state=42)

    print("\n📊 Distribución final de clases:")
    print(data_balanced['label_desc'].value_counts())

    # === Guardado en formato eficiente ===
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    parquet_path = "../data/processed/balanced_data.parquet"
    data_balanced.to_parquet(parquet_path, index=False)
    print(f"\n✅ Dataset procesado guardado en formato Parquet: {parquet_path}")

# =====================================================
# EJECUCIÓN DIRECTA
# =====================================================
if __name__ == "__main__":
    process_data()
