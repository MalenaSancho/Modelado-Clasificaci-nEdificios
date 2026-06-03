import os
import pickle
import pandas as pd
import json
from sklearn import metrics
from collections import Counter

in_path = "../out/comparison" # Cargamos el resultado DEL PASO 02
out_path = "../out2/comparison2"

# 1. Creamos el directorio usando Python nativo (sin necesidad de utils)
os.makedirs(out_path, exist_ok=True)

print("Cargando las intersecciones precalculadas del paso 02...")
with open(os.path.join(in_path, 'name2intersection.pickle'), 'rb') as handle:
    name2intersection = pickle.load(handle)

try:
    with open("diccionario_etiquetas_llm.json", "r", encoding="utf-8") as f:
        diccionario_llm = json.load(f)
except FileNotFoundError:
    print("CUIDADO: No se encontró 'diccionario_etiquetas_llm.json'.")
    diccionario_llm = {}

name2performance_new = dict()

for name, intersection in name2intersection.items():
    print(f"\nAplicando lógica mejorada para: {name}...")
    
    # 2. Guardamos la predicción original y calculamos área
    intersection['type_original'] = intersection['type']
    intersection['area_m2'] = intersection.geometry.area

    # 3. Función inteligente de decisión
    def mejorar_clasificacion(row):
        tag = row['tag used']
        area = row['area_m2']
        tipo_original = row['type_original']

        if pd.isna(tag):
            return 'NON_RES' if area > 800 else 'RES'

        clasificacion_llm = diccionario_llm.get(tag, "N/A")
        if clasificacion_llm in ["RES", "NON_RES"]:
            return clasificacion_llm
        else:
            return tipo_original

    # 4. Aplicar nueva lógica
    intersection['type'] = intersection.apply(mejorar_clasificacion, axis=1)

    # 5. Recalcular métricas
    official_values = intersection['org_type']
    predicted = intersection['type']
    performance = metrics.classification_report(official_values, predicted)
    
    name2performance_new[name] = performance
    print(performance)
    
    # Actualizar diccionario
    name2intersection[name] = intersection

# Guardar los nuevos resultados mejorados
with open(os.path.join(out_path, 'name2performance.pickle'), 'wb') as handle:
    pickle.dump(name2performance_new, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(out_path, 'name2intersection.pickle'), 'wb') as handle:
    pickle.dump(name2intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\n¡Proceso instantáneo completado! Resultados guardados en {out_path}")