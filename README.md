# 🌌 Astrolab-IA

**Astrolab-IA** is an **AI-powered platform** that helps scientists and enthusiasts explore, classify, and prioritize exoplanets discovered by NASA missions.  
By combining open datasets from **Kepler**, **K2**, and **TESS**, Astrolab-IA uses machine learning to automatically classify observed objects as **Confirmed Planets**, **Candidates**, or **False Positives**, while providing **explainable insights** into each prediction.

---

## Data Sources

Astrolab-IA integrates **open data** from NASA’s missions:

| Mission     | Dataset                           | Main Label Field   | Description                                  |
|--------------|------------------------------------|--------------------|----------------------------------------------|
| **Kepler**   | Kepler Objects of Interest (KOI)  | `koi_disposition`  | Confirmed, Candidate, or False Positive      |
| **K2**       | K2 Planets and Candidates         | `disposition`      | Archive Disposition (CONFIRMED, CANDIDATE…)  |
| **TESS**     | TESS Objects of Interest (TOI)    | `tfopwg_disp`      | CP, PC, FP, FA, KP, APC                      |

All datasets were **cleaned, standardized, and merged** into a unified table of physical parameters to allow cross-mission comparison.

---

## Model Pipeline

### Input Features (common across all missions)
- Orbital period, transit duration, and transit depth  
- Planetary radius, equilibrium temperature, and insolation  
- Stellar radius, surface gravity, and temperature  
- RA and DEC (used for grouping, not for training)

### Training Setup
- AutoML (**FLAML**) using LightGBM, XGBoost, Random Forest, and Logistic Regression  
- Stratified train/test split with **macro-F1** as the main metric  
- Explainability via **SHAP values** and **Permutation Importance**  
- An optional **Fuzzy Logic layer** estimates a *habitability score* by comparing planetary features to Earth’s conditions

---

## Web Application

The web platform allows users to:

- View **model statistics** such as accuracy, F1-score, and confusion matrix  
- **Explore** and visualize confirmed planets and promising candidates  
- **Predict** the classification of new exoplanet candidates using model inference  
- Access **explainability tools** to understand each prediction  

Built with **HTML, CSS, and JavaScript**, the web interface focuses on accessibility and free visual storytelling.

---

### 🛰️ *Astrolab-IA — Team DataVerse*

This project uses **open NASA data**.

---------------------------------------------------------------------

# 🌌 Astrolab-IA

**Astrolab-IA** es una **plataforma impulsada por inteligencia artificial** que ayuda a científicos y entusiastas a explorar, clasificar y priorizar exoplanetas descubiertos por las misiones de la NASA.  
Al combinar conjuntos de datos abiertos de **Kepler**, **K2** y **TESS**, Astrolab-IA utiliza aprendizaje automático para clasificar automáticamente los objetos observados como **Planetas Confirmados**, **Candidatos** o **Falsos Positivos**, proporcionando además **explicaciones interpretables** para cada predicción.

---

## Fuentes de Datos

Astrolab-IA integra **datos abiertos** de las misiones de la NASA:

| Misión       | Conjunto de Datos                 | Campo Principal    | Descripción                                 |
|---------------|----------------------------------|--------------------|---------------------------------------------|
| **Kepler**    | Objetos de Interés de Kepler (KOI) | `koi_disposition` | Confirmado, Candidato o Falso Positivo      |
| **K2**        | Planetas y Candidatos K2         | `disposition`      | Disposición en archivo (CONFIRMED, CANDIDATE…) |
| **TESS**      | Objetos de Interés de TESS (TOI) | `tfopwg_disp`      | CP, PC, FP, FA, KP, APC                     |

Todos los conjuntos de datos fueron **limpiados, estandarizados y combinados** en una tabla unificada de parámetros físicos para permitir la comparación entre misiones.

---

## Flujo del Modelo

### Características de Entrada (comunes en todas las misiones)
- Período orbital, duración del tránsito y profundidad del tránsito  
- Radio planetario, temperatura de equilibrio e insolación  
- Radio estelar, gravedad superficial y temperatura  
- RA y DEC (utilizados para agrupar, no para el entrenamiento)

### Configuración del Entrenamiento
- AutoML (**FLAML**) usando LightGBM, XGBoost, Random Forest y Regresión Logística  
- División estratificada de entrenamiento/prueba con **macro-F1** como métrica principal  
- Explicabilidad mediante **valores SHAP** e **Importancia por Permutación**  
- Una capa opcional de **Lógica Difusa** estima un *índice de habitabilidad* comparando las características planetarias con las condiciones de la Tierra

---

## Aplicación Web

La plataforma web permite a los usuarios:

- Ver **estadísticas del modelo** como precisión, F1-score y matriz de confusión  
- **Explorar** y visualizar planetas confirmados y candidatos prometedores  
- **Predecir** la clasificación de nuevos candidatos a exoplanetas mediante la inferencia del modelo  
- Acceder a **herramientas de explicabilidad** para comprender cada predicción  

Construida con **HTML, CSS y JavaScript**, la interfaz web se centra en la accesibilidad y en una narrativa visual libre e intuitiva.

---

### 🛰️ *Astrolab-IA — Equipo DataVerse*

Este proyecto utiliza **datos abiertos de la NASA**.
