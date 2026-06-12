# Clasificación de Emociones — FER2013
### Trabajo Práctico Final · Procesamiento de Imágenes y Señales · UTN

Pipeline completo de clasificación de emociones faciales comparando Deep Learning y Machine Learning clásico sobre el dataset **FER2013** (35,887 imágenes de rostros, 7 emociones).

---

## Estructura del proyecto

```
TP/
├── src/                          # Notebooks ejecutables
│   ├── TP_Final_Paso1_EDA_Preprocesamiento.ipynb
│   ├── TP_Final_Paso2_DeepLearning.ipynb
│   ├── TP_Final_Paso3_ML_Clasico_XGBoost.ipynb
│   ├── TP_Final_Paso4_Evaluacion_Comparacion.ipynb
│   └── TP_Final_Paso5_Interpretabilidad.ipynb
└── html/                         # Exports HTML (visualización sin instalar nada)
    ├── TP_Final_Paso1_EDA_Preprocesamiento.html
    ├── TP_Final_Paso2_DeepLearning.html
    ├── TP_Final_Paso3_ML_Clasico_XGBoost.html
    ├── TP_Final_Paso4_Evaluacion_Comparacion.html
    └── TP_Final_Paso5_Interpretabilidad.html
```

---

## Pipeline

| Paso | Notebook | Contenido |
|------|----------|-----------|
| 1 | EDA y Preprocesamiento | Carga, análisis de balanceo, FFT 2D, filtros Gaussiano/Sobel, Data Augmentation, normalización Z-Score |
| 2 | Deep Learning | MiniEmotionNet (CNN custom) + ResNet18 Transfer Learning, AdamW, Early Stopping, extracción de features |
| 3 | ML Clásico | XGBoost sobre features del encoder CNN/ResNet, RFECV para selección de features |
| 4 | Evaluación | Curvas ROC-AUC por clase, matrices de confusión, radar F1, tabla comparativa |
| 5 | Interpretabilidad | SHAP Values sobre XGBoost, Grad-CAM implementado from scratch sobre la CNN |

---

## Dataset

**FER2013** — Facial Expression Recognition 2013  
7 clases: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`  
Train: 28,709 imágenes · Test: 7,178 imágenes · Resolución: 48×48 px (escala de grises)

> El dataset no está incluido en el repositorio. Descargar desde [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013) y ubicar en `../Dataset/train/` y `../Dataset/test/`.

---

## Instalación

```bash
pip install numpy pandas matplotlib seaborn opencv-python Pillow scikit-learn albumentations tqdm torch torchvision xgboost shap nbconvert
```

## Ejecución

Correr en orden: Paso 1 → 2 → 3 → 4 → 5. Cada paso guarda artefactos (`.npy`, `.csv`, `.pt`) que el siguiente consume.

---

## Conexión con los contenidos de la materia

| Concepto (PIYS UTN) | Aplicación en el TP |
|---|---|
| Convolución 2D | Operación central de cada capa de la CNN |
| Filtrado pasa-bajos/altos | Gaussiano (suavizado) y Sobel (bordes) como preprocesamiento EDA |
| Dominio frecuencial (FFT) | Análisis del espectro de magnitud promedio por emoción |
| Correlación cruzada | Grad-CAM usa gradientes × activaciones (correlación entre mapas) |
| Media Móvil | Equivalente 2D: filtro Gaussiano como suavizado espacial |
