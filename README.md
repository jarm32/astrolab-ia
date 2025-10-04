# 🌌 Astrolab-IA

**Astrolab-IA** is an **AI-powered platform** that helps scientists and enthusiasts explore, classify, and prioritize exoplanets discovered by NASA missions.  
By combining open datasets from **Kepler**, **K2**, and **TESS**, Astrolab-IA uses machine learning to automatically classify observed objects as **Confirmed Planets**, **Candidates**, or **False Positives**, while providing **explainable insights** into each prediction.

---

## 🧩 Data Sources

Astrolab-IA integrates **open data** from NASA’s missions:

| Mission     | Dataset                           | Main Label Field   | Description                                  |
|--------------|------------------------------------|--------------------|----------------------------------------------|
| **Kepler**   | Kepler Objects of Interest (KOI)  | `koi_disposition`  | Confirmed, Candidate, or False Positive      |
| **K2**       | K2 Planets and Candidates         | `disposition`      | Archive Disposition (CONFIRMED, CANDIDATE…)  |
| **TESS**     | TESS Objects of Interest (TOI)    | `tfopwg_disp`      | CP, PC, FP, FA, KP, APC                      |

All datasets were **cleaned, standardized, and merged** into a unified table of physical parameters to allow cross-mission comparison.

---

## 🧮 Model Pipeline

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

## 💻 Web Application

The web platform allows users to:

- 📊 View **model statistics** such as accuracy, F1-score, and confusion matrix  
- 🔎 **Explore** and visualize confirmed planets and promising candidates  
- 🌠 **Predict** the classification of new exoplanet candidates using model inference  
- 💬 Access **explainability tools** to understand each prediction  

Built with **HTML, CSS, and JavaScript**, the web interface focuses on accessibility and free visual storytelling.

---

### 🛰️ *Astrolab-IA — Team DataVerse*

This project uses **open NASA data**.
