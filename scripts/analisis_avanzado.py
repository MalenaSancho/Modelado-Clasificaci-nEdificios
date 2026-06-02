import pickle
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Ruta donde tienes tus archivos pickle guardados (ajústala si es necesario)
input_path = "../out/comparison"
output_path = "../out/graficas_memoria" # Carpeta para guardar las imágenes

# Crear la carpeta de salida si no existe
if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Cargando datos ya procesados (esto tardará solo unos segundos)...")
with open(os.path.join(input_path, 'name2intersection.pickle'), 'rb') as handle:
    name2intersection = pickle.load(handle)

for name, intersection in name2intersection.items():
    print(f"\n{'='*50}\nAnálisis para: {name}\n{'='*50}")

    official_values = intersection['org_type']
    predicted = intersection['type']

    # 1. Reporte de Métricas
    print("MÉTRICAS DE CLASIFICACIÓN:")
    print(metrics.classification_report(official_values, predicted))

    # 2. Generación y Guardado de la Matriz de Confusión
    plt.figure(figsize=(6, 4))
    cm = metrics.confusion_matrix(official_values, predicted, labels=["RES", "NON_RES"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["RES", "NON_RES"], 
                yticklabels=["RES", "NON_RES"])
    
    plt.title(f'Matriz de Confusión - {name}')
    plt.xlabel('Predicción de OSM (Nuestro Modelo)')
    plt.ylabel('Ground Truth (Dato Oficial)')
    plt.tight_layout()
    
    # Guardamos la imagen para que la puedas pegar directamente en el Word de la memoria
    imagen_filename = os.path.join(output_path, f"{name}_matriz_confusion.png")
    plt.savefig(imagen_filename, dpi=300)
    print(f" -> Gráfica guardada en: {imagen_filename}")
    plt.close() # Cierra la gráfica para que no se acumulen en memoria

    # 3. Análisis de Errores Críticos (Para la memoria)
    gdf_error = intersection[intersection['type'] != intersection['org_type']]
    
    print("\n--- ANÁLISIS DE ERRORES ---")
    print("Top 5 Etiquetas de OSM que causaron falsos RESIDENCIALES (Era NON_RES pero se clasificó como RES):")
    errores_res = Counter(gdf_error[gdf_error['type']=='RES']['tag used']).most_common(5)
    for tag, freq in errores_res:
        print(f"  - {tag}: {freq} veces")
        
    print("\nTop 5 Etiquetas de OSM que causaron falsos NO RESIDENCIALES (Era RES pero se clasificó como NON_RES):")
    errores_non_res = Counter(gdf_error[gdf_error['type']=='NON_RES']['tag used']).most_common(5)
    for tag, freq in errores_non_res:
        print(f"  - {tag}: {freq} veces")