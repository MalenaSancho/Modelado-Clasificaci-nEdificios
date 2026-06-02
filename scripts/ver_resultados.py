import pickle
import os
import pandas as pd
import geopandas as gpd

# Rutas de entrada y salida
input_path = "../out/comparison"
output_path = "../out/comparison_transformados"

# Asegurarnos de que la carpeta de salida existe (si no, la crea)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Carpeta creada: {output_path}")

# 1. CARGAR Y VER LAS MÉTRICAS DE RENDIMIENTO
print("\nCargando métricas de rendimiento...\n")
with open(os.path.join(input_path, 'name2performance.pickle'), 'rb') as handle:
    name2performance = pickle.load(handle)

for condado, reporte in name2performance.items():
    print(f"=====================================================")
    print(f"RESULTADOS PARA: {condado}")
    print(f"=====================================================")
    print(reporte)
    print("\n")


# 2. CARGAR LAS TABLAS DE EDIFICIOS Y EXPORTARLAS A CSV (EXCEL)
print("Exportando datos de edificios a CSV...")
with open(os.path.join(input_path, 'name2intersection.pickle'), 'rb') as handle:
    name2intersection = pickle.load(handle)

for condado, gdf in name2intersection.items():
    # Convertimos el GeoDataFrame a un DataFrame normal de Pandas
    # Eliminamos la columna de geometría (polígonos) porque Excel no la entiende bien
    df = pd.DataFrame(gdf.drop(columns=['geometry']))
    
    # Guardamos el archivo CSV en la NUEVA carpeta
    csv_filename = os.path.join(output_path, f"{condado}_resultados_completos.csv")
    df.to_csv(csv_filename, index=False)
    print(f" -> Guardado con éxito: {csv_filename}")


# 3. EXTRAER LOS MAPAS GEOMÉTRICOS COMPLETOS
print("\nExportando mapas completos a GeoJSON (ideal para QGIS/ArcGIS)...")
with open(os.path.join(input_path, 'name2identified_gdf.pickle'), 'rb') as handle:
    name2identified_gdf = pickle.load(handle)

for condado, gdf in name2identified_gdf.items():
    # El formato GeoJSON es moderno, seguro y perfecto para mapas web y QGIS
    # Lo guardamos en la NUEVA carpeta
    mapa_filename = os.path.join(output_path, f"{condado}_mapa_completo.geojson")
    
    # Exportamos el archivo
    gdf.to_file(mapa_filename, driver='GeoJSON')
    print(f" -> Mapa guardado con éxito: {mapa_filename}")

print(f"\n¡Proceso terminado! Todos tus archivos transformados están listos en: {output_path}")