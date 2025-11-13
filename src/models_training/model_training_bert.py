"""
model_training_bert.py
----------------------------------
Entrena un modelo BERT fine-tuned para la clasificación de texto
en tres categorías (Hate Speech, Offensive Language, Neither).

Entradas:
    - data/processed/balanced_data.parquet

Salidas:
    - Modelo entrenado y tokenizer: data/models/BERTmodel_training/
    - Estados del optimizador: data/models/BERTmodel_training/training_states.pt

Autor: Vivian Garavito
Fecha: 11/11/2025
"""

# =====================================================
# === Librerías Base ===
# =====================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# === PyTorch y Transformers ===
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# =====================================================
# === Configuración Inicial ===
# =====================================================
DATA_PATH = "../data/processed/balanced_data.parquet"
MODEL_PATH = "../data/models/BeRT_Training/"

# Crear carpeta destino del modelo
os.makedirs(MODEL_PATH, exist_ok=True)

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Dispositivo activo: {device}")

# =====================================================
# === Carga del Dataset Procesado ===
# =====================================================
data = pd.read_parquet(DATA_PATH)
print(f"Dataset cargado correctamente. Total de registros: {len(data)}")

# División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    data['tweet'],
    data['label'],
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)
print(f"Split -> Train: {len(X_train)}, Test: {len(X_test)}")

# =====================================================
# === Tokenización con BERT ===
# =====================================================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 128

def prepare_data(texts, labels, tokenizer, max_len=128):
    """Tokeniza y genera tensores de entrada para BERT."""
    input_ids, attention_masks = [], []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.values)
    return input_ids, attention_masks, labels

train_input_ids, train_attention_masks, train_labels = prepare_data(X_train, y_train, tokenizer, MAX_LEN)
test_input_ids, test_attention_masks, test_labels = prepare_data(X_test, y_test, tokenizer, MAX_LEN)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# DataLoaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
validation_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# =====================================================
# === Modelo BERT Fine-Tuned ===
# =====================================================
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)
print("Modelo BERT cargado y preparado para entrenamiento.")

# =====================================================
# === Entrenamiento ===
# =====================================================
epochs = 3
optimizer = AdamW(model.parameters(), lr=3e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = CrossEntropyLoss()

for epoch in range(epochs):
    print(f"\n======== Epoch {epoch + 1} / {epochs} ========")
    print("Entrenando...")
    model.train()
    total_train_loss = 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Batches")):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # Validación
    model.eval()
    total_eval_loss, total_eval_acc = 0, 0
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits

        total_eval_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        total_eval_acc += (preds == b_labels).cpu().numpy().mean()

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    avg_val_acc = total_eval_acc / len(validation_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {avg_val_acc:.4f}")

print("\nEntrenamiento completado con éxito.")

# =====================================================
# === Guardar Modelo y Tokenizer ===
# =====================================================
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
print(f" Modelo y tokenizer guardados en: {MODEL_PATH}")

# Guardar estados del optimizador y scheduler
torch.save({
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
}, os.path.join(MODEL_PATH, "training_states.pt"))

print("Estados de entrenamiento guardados.")
