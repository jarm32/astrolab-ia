# === Recorre CSV + JSON y escribe el score difuso al JSON (rápido + feedback) ===
import json
import os
import time

import pandas as pd
import numpy as np

# 1) Trae la función definir_variables
try:
    from logicaDifusa import definir_variables as fuzzy_score  # si la tienes en un archivo
except Exception:
    from __main__ import definir_variables as fuzzy_score       # si está en una celda del notebook

CSV_PATH  = "exoplanetas_unificado.csv"
JSON_IN   = "exoplanetas_light.json"
JSON_OUT  = "exoplanetas_light_scored.json"
JSON_TMP  = JSON_OUT + ".tmp"   # guardado incremental

# --------- Parámetros de ejecución ----------
N_SAVE = 500        # guarda cada N registros procesados
PRINT_EVERY = 250   # imprime un mini-resumen cada N
USE_TQDM = True     # intenta usar tqdm para progreso bonito
# -------------------------------------------

# 2) Carga CSV (llave = object_id)
t0 = time.time()
df = pd.read_csv(CSV_PATH, low_memory=False)
if "object_id" not in df.columns:
    raise ValueError("El CSV no tiene columna 'object_id' (necesaria para el enlace).")

df["object_id"] = df["object_id"].astype(str)
df = df.set_index("object_id", drop=False)

# Para acelerar búsquedas: nos quedamos sólo con columnas necesarias
NEEDED = {
    "pl_radio", "pl_temperatura_eq", "insolacion", "periodo_orbital",
    "st_temperatura", "st_radio", "st_gravedad", "object_id"
}
present_cols = [c for c in df.columns if c in NEEDED]
df_small = df[present_cols].copy()

# --- Eliminar duplicados de object_id, quedarnos con la primera fila ---
dup_count = df_small.index.duplicated(keep=False).sum()
if dup_count:
    uniq_dups = df_small.index[df_small.index.duplicated(keep=False)].unique()
    print(f"[WARN] Se encontraron {dup_count} registros duplicados "
          f"({len(uniq_dups)} IDs distintos). Ejemplos: {list(uniq_dups[:10])}")
df_small = df_small[~df_small.index.duplicated(keep="first")]

# Convertimos a dict para acceso rápido
df_dict = df_small.to_dict(orient="index")
del df, df_small  # liberar memoria

# 3) Carga JSON
with open(JSON_IN, "r", encoding="utf-8") as f:
    records = json.load(f)

# 4) Mapeos
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
    """Prefiere JSON > CSV > default."""
    json_col = MAP_JSON.get(key)
    csv_col  = MAP_CSV.get(key)

    if json_col and (json_col in rec):
        v = _to_float(rec.get(json_col))
        if v is not None:
            return v

    if rowdict is not None and csv_col:
        v = _to_float(rowdict.get(csv_col))
        if v is not None:
            return v

    return float(default)

# --- Reanudación: si existe un TMP con progreso previo, lo cargamos ---
already = 0
if os.path.exists(JSON_TMP):
    try:
        with open(JSON_TMP, "r", encoding="utf-8") as f:
            partial = json.load(f)
        done_ids = {str(r.get("object_id")) for r in partial if r.get("earth_similarity", None) is not None}
        id_to_partial = {str(r.get("object_id")): r for r in partial}
        new_records = []
        for rec in records:
            oid = str(rec.get("object_id", ""))
            if oid in id_to_partial:
                new_records.append(id_to_partial[oid])
            else:
                new_records.append(rec)
        records = new_records
        already = len([r for r in records if r.get("earth_similarity", None) is not None])
        print(f"↩ Reanudando desde {JSON_TMP}. Registros ya puntuados: {already}")
    except Exception as e:
        print(f"[WARN] No se pudo reanudar desde {JSON_TMP}: {e}")

total = len(records)
procesados, sin_csv, sin_entry = 0, 0, 0
faltantes_csv = []
last_save = 0
last_print = 0

# tqdm opcional
pbar = None
if USE_TQDM:
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, initial=already, desc="Calculando fuzzy score", unit="obj")
    except Exception:
        pbar = None

def safe_fuzzy(entrada, oid):
    try:
        score, _cats = fuzzy_score(entrada)
        return float(round(score, 1)), None
    except Exception as e:
        return None, f"[WARN] Fallo al calcular fuzzy para {oid}: {e}"

start = time.time()

for i, rec in enumerate(records):
    if rec.get("earth_similarity", None) is not None:
        if pbar: pbar.update(1)
        continue

    label = str(rec.get("label", "")).upper()
    if label not in ("CONFIRMED", "CANDIDATE"):
        rec["earth_similarity"] = None
        if pbar: pbar.update(1)
        continue

    oid = str(rec.get("object_id", ""))
    if not oid:
        sin_entry += 1
        rec["earth_similarity"] = None
        if pbar: pbar.update(1)
        continue

    rowdict = df_dict.get(oid)
    if rowdict is None:
        sin_csv += 1
        faltantes_csv.append(oid)

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
    rec["earth_similarity"] = score
    if err:
        print(err)

    procesados += 1
    if pbar: pbar.update(1)

    if not pbar and procesados - last_print >= PRINT_EVERY:
        elapsed = time.time() - start
        done = already + procesados
        rate = done / max(elapsed, 1e-9)
        remaining = total - done
        eta = remaining / max(rate, 1e-9)
        print(f"[{done}/{total}] {rate:.1f} obj/s | ETA ~ {eta/60:.1f} min")
        last_print = procesados

    if procesados - last_save >= N_SAVE:
        with open(JSON_TMP, "w", encoding="utf-8") as ftmp:
            json.dump(records, ftmp, ensure_ascii=False, indent=2)
        last_save = procesados
        if not pbar:
            print(f"💾 Guardado incremental: {already + procesados}/{total}")

# 6) Guarda JSON final
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

try:
    if os.path.exists(JSON_TMP):
        os.remove(JSON_TMP)
except Exception:
    pass

if pbar: pbar.close()

elapsed_total = time.time() - t0
print(f"✔ Scores escritos / actualizados para {procesados} objetos (CONFIRMED/CANDIDATE).")
print(f"ℹ Sin fila en CSV: {sin_csv} | Registros sin object_id: {sin_entry}")
if faltantes_csv[:10]:
    print("Ejemplos sin CSV:", faltantes_csv[:10])
print(f"Salida: {JSON_OUT}")
print(f"⏱ Tiempo total: {elapsed_total:.1f} s")
