# === Imputa CSV + recalcula earth_similarity nulo en JSON con lógica difusa (con progreso + guardado incremental) ===
import json
import os
import time

import pandas as pd
import numpy as np

# 1) Importa la función de lógica difusa
try:
    from logicaDifusa import definir_variables as fuzzy_score  # si está en un .py
except Exception:
    from __main__ import definir_variables as fuzzy_score       # si está en una celda del notebook

# ---- Rutas (ajústalas si ejecutas fuera del entorno actual) ----
CSV_PATH   = "exoplanetas_unificado.csv"
JSON_IN    = "exoplanetas_light_scored.json"
JSON_OUT   = "exoplanetas_light_scored_imputed.json"
JSON_TMP   = JSON_OUT + ".tmp"   # guardado incremental / reanudación

# ---- Parámetros de ejecución ----
USE_TQDM      = True   # intenta usar tqdm para una barra chula
PRINT_EVERY   = 500    # fallback: imprime mini-resumen cada N
SAVE_EVERY    = 1000   # guarda incremental cada N procesados nuevos
ROUND_SCORE_1D = True  # redondea el score a 1 decimal
# -------------------------------

t0 = time.time()

# 2) Carga CSV y deja indexado por object_id
df = pd.read_csv(CSV_PATH, low_memory=False)

if "object_id" not in df.columns:
    raise ValueError("El CSV no tiene columna 'object_id' (necesaria para el enlace).")

# normaliza object_id a str e indexa
df["object_id"] = df["object_id"].astype(str)
df = df.set_index("object_id", drop=False)

# Columnas que nos interesan desde CSV (para imputation)
NEEDED = [
    "pl_radio", "pl_temperatura_eq", "insolacion", "periodo_orbital",
    "st_temperatura", "st_radio", "st_gravedad"
]
present_cols = [c for c in NEEDED + ["object_id"] if c in df.columns]
df = df[present_cols].copy()

# 3) Elimina duplicados por object_id (conserva la primera fila)
dup_count = df.index.duplicated(keep=False).sum()
if dup_count:
    ids_dup = df.index[df.index.duplicated(keep=False)].unique()[:10]
    print(f"[WARN] Duplicados en CSV: {dup_count} filas. Ejemplos de IDs: {list(ids_dup)}")
df = df[~df.index.duplicated(keep="first")]

# 4) Imputación: convierte a numérico lo que aplique y rellena NaN con la mediana de cada columna
for col in NEEDED:
    if col not in df.columns:
        # si falta por completo, créala como NaN para luego usar defaults
        df[col] = np.nan
    # intenta convertir a float (si viene como texto)
    df[col] = pd.to_numeric(df[col], errors="coerce")

# calcula medianas y rellena NaN
medianas = {col: df[col].median(skipna=True) for col in NEEDED}
df[NEEDED] = df[NEEDED].fillna(value=medianas)

# 5) Acceso rápido: pasa CSV imputado a dict {object_id: {col: val}}
df_dict = df.to_dict(orient="index")
del df  # libera memoria

# 6) Carga JSON
with open(JSON_IN, "r", encoding="utf-8") as f:
    records = json.load(f)

# 7) Mapeos JSON/CSV → entradas del sistema difuso
MAP_CSV = {
    "radius":  "pl_radio",
    "teq":     "pl_temperatura_eq",
    "insol":   "insolacion",
    "period":  "periodo_orbital",
    "st_teff": "st_temperatura",
    "st_rad":  "st_radio",
    "st_logg": "st_gravedad",
}
MAP_JSON = {
    "radius": "pl_radio",
    "teq":    "pl_temperatura_eq",
    "period": "periodo_orbital",
}

def _to_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None

def pick_value(rowdict, rec, key, default):
    """
    Prefiere JSON > CSV_imputado > default.
    rowdict: dict de columnas CSV para ese object_id (ya imputado). Puede ser None si no existe.
    """
    json_col = MAP_JSON.get(key)
    csv_col  = MAP_CSV.get(key)

    # JSON primero
    if json_col and (json_col in rec):
        v = _to_float(rec.get(json_col))
        if v is not None:
            return v

    # CSV imputado después
    if rowdict is not None and csv_col:
        v = _to_float(rowdict.get(csv_col))
        if v is not None:
            return v

    # Default
    return float(default)

# 8) Reanudación (si existe TMP lo usamos como base de trabajo)
already = 0
if os.path.exists(JSON_TMP):
    try:
        with open(JSON_TMP, "r", encoding="utf-8") as f:
            partial = json.load(f)
        # combinamos por object_id dando preferencia a lo ya guardado en tmp
        by_id_tmp = {str(r.get("object_id")): r for r in partial}
        new_records = []
        for rec in records:
            oid = str(rec.get("object_id", ""))
            new_records.append(by_id_tmp.get(oid, rec))
        records = new_records
        already = sum(1 for r in records if r.get("earth_similarity") is not None)
        print(f"↩ Reanudando desde TMP. Ya calculados: {already}")
    except Exception as e:
        print(f"[WARN] No se pudo reanudar desde TMP: {e}")

# 9) Progreso (tqdm si disponible)
pbar = None
if USE_TQDM:
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(records), initial=0, desc="Actualizando JSON", unit="obj")
    except Exception:
        pbar = None

# 10) Bucle principal: solo recalculamos donde earth_similarity es None y label != FALSE POSITIVE
procesados = 0
saltados_fp = 0
saltados_ya_tenian_score = 0
sin_entry = 0
sin_csv = 0
faltantes_csv = []

last_print = 0
last_save = 0
start = time.time()

def safe_fuzzy(entrada, oid):
    try:
        score, _cats = fuzzy_score(entrada)
        if ROUND_SCORE_1D:
            score = float(round(score, 1))
        else:
            score = float(score)
        return score, None
    except Exception as e:
        return None, f"[WARN] Fallo fuzzy para {oid}: {e}"

total = len(records)

for idx, rec in enumerate(records):
    if pbar: pbar.update(1)

    label = str(rec.get("label", "")).upper()
    oid = str(rec.get("object_id", "") or "")
    existing = rec.get("earth_similarity", None)

    # Si ya tiene score, no lo tocamos
    if existing is not None:
        saltados_ya_tenian_score += 1
        # feed opcional
        continue

    # En falsos positivos NO se calcula (se deja None)
    if label == "FALSE POSITIVE":
        saltados_fp += 1
        continue

    if not oid:
        sin_entry += 1
        continue

    rowdict = df_dict.get(oid)
    if rowdict is None:
        sin_csv += 1
        faltantes_csv.append(oid)

    # Construye la entrada de la lógica difusa, con CSV ya imputado
    entrada = {
        "radius":  pick_value(rowdict, rec, "radius", 1.0),
        "teq":     pick_value(rowdict, rec, "teq", 290.0),
        "insol":   pick_value(rowdict, rec, "insol", 1.0),
        "period":  pick_value(rowdict, rec, "period", 50.0),
        "st_teff": pick_value(rowdict, rec, "st_teff", 5777.0),
        "st_rad":  pick_value(rowdict, rec, "st_rad", 1.0),
        "st_logg": pick_value(rowdict, rec, "st_logg", 4.4),
    }

    score, err = safe_fuzzy(entrada, oid)
    rec["earth_similarity"] = score  # si falló, quedará None
    if err:
        print(err)
    procesados += 1

    # Fallback de progreso si no hay tqdm: imprime cada PRINT_EVERY
    if not pbar and (idx + 1) - last_print >= PRINT_EVERY:
        elapsed = time.time() - start
        done = idx + 1
        rate = done / max(elapsed, 1e-9)
        remaining = total - done
        eta = remaining / max(rate, 1e-9)
        print(f"[{done}/{total}] {rate:.1f} obj/s | ETA ~ {eta/60:.1f} min")
        last_print = idx + 1

    # Guardado incremental
    if procesados - last_save >= SAVE_EVERY:
        with open(JSON_TMP, "w", encoding="utf-8") as ftmp:
            json.dump(records, ftmp, ensure_ascii=False, indent=2)
        last_save = procesados

# 11) Guarda JSON final y limpia TMP
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

try:
    if os.path.exists(JSON_TMP):
        os.remove(JSON_TMP)
except Exception:
    pass

if pbar: pbar.close()

elapsed_total = time.time() - t0
print("===============================================")
print(f"✔ Recalculados (earth_similarity era null): {procesados}")
print(f"• Saltados (label=FALSE POSITIVE): {saltados_fp}")
print(f"• Saltados (ya tenían earth_similarity): {saltados_ya_tenian_score}")
print(f"• Registros sin object_id: {sin_entry}")
print(f"• Registros sin fila en CSV: {sin_csv}")
if faltantes_csv[:10]:
    print("Ejemplos sin CSV:", faltantes_csv[:10])
print(f"Salida final: {JSON_OUT}")
print(f"⏱ Tiempo total: {elapsed_total:.1f} s")

