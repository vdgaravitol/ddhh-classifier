# ddhh classifier
#  Clasificador de Texto para Protección de Derechos Humanos

Este proyecto implementa un **sistema de clasificación de texto** que identifica posibles **vulneraciones a los Derechos Humanos (DDHH)** en mensajes digitales.  
Utiliza un esquema tipo **semáforo**:

- 🟢 **VERDE:** Contenido normal  
- 🟡 **AMARILLO:** Lenguaje potencialmente problemático  
- 🔴 **ROJO:** Contenido que vulnera derechos (odio, amenazas, discriminación)

---

##  Modelos desarrollados

1. **TF-IDF + Random Forest:**  
   Modelo tradicional basado en frecuencia de palabras. Rápido, interpretable y liviano.

2. **Fine-Tuned BERT:**  
   Modelo Transformer ajustado para clasificación de discurso en inglés, con mayor sensibilidad contextual y semántica.

---

##  Estructura del proyecto

ddhh-classifier/
├── data/
│ ├── raw/ # Datos originales
│ ├── processed/ # Datos procesados
│ └── models/ # Modelos entrenados (no incluidos por tamaño)
├── notebooks/
│ ├── exploration.ipynb # Análisis exploratorio
│ └── test.ipynb # Comparación de inferencia
├── src/
│ ├── Inference/
│ │ ├── inference_bert.py
│ │ └── inference_tfidf_rf.py
│ ├── models_training/
│ │ ├── model_training_bert.py
│ │ └── model_training_TF-IDF_RF.py
│ ├── data_preprocessing.py
├── tests/
│ └── test.py
├── requirements.txt
├── Dockerfile
└── README.md


---

## ⚙️ Instalación del entorno

```bash
# Clonar el repositorio
git clone https://github.com/vdgaravitol/ddhh-classifier.git
cd ddhh-classifier

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate   # Windows
# o
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt


#Uso de los modelos
# Inferencia con BERT
python src/Inference/inference_bert.py

# Inferencia con TF-IDF + Random Forest
python src/Inference/inference_tfidf_rf.py

Tras comparar ambos enfoques, el modelo BERT fine-tuned fue seleccionado como modelo principal por su mayor sensibilidad semántica y capacidad para identificar lenguaje ambiguo o implícitamente violento.
Esto lo hace más adecuado en contextos de Protección de Derechos Humanos, donde los falsos negativos (mensajes dañinos no detectados) tienen un costo ético elevado.

El modelo TF-IDF + Random Forest se conserva como baseline liviano y explicable, útil para auditorías o entornos con menos recursos.

✅ Modelo final elegido: Fine-Tuned BERT
🎯 Justificación: mejor contexto lingüístico, mayor recall y precisión ética.

⚖️ Consideraciones éticas

Los modelos pueden reflejar sesgos del dataset de origen.

Deben usarse como herramientas de apoyo, no como reemplazo de evaluación humana.

