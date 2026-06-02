import os
import sys
homedir = "../"
codedir = os.path.join(homedir, 'src')
sys.path.append(codedir)
import map_buildings, utils
from shapely.geometry import MultiPolygon, Polygon
import pickle
import osmnx as ox
from sklearn import metrics
from collections import Counter
from tqdm import tqdm
import pandas as pd
import json # <-- AÑADIDO: Importación necesaria para leer el diccionario del LLM

tqdm.pandas()

in_path = "../data/"
out_path = "../out2/"
cache_folder = "../osmnx_cache2/"

ox.settings.cache_folder = cache_folder
ox.settings.requests_timeout = 500

def get_largest_overlap(row, gdf_official):
    overlaps = gdf_official[gdf_official.intersects(row.geometry)]
    if overlaps.empty:
        return None
    largest_overlap_index = overlaps.apply(lambda x: x.geometry.intersection(row.geometry).area, axis=1).idxmax()
    
    return gdf_official.loc[largest_overlap_index]['org_type'], gdf_official.loc[largest_overlap_index]['old_label']

with open(os.path.join(in_path, 'test_region2gdf.pickle'), 'rb') as handle:
    name2gdf = pickle.load(handle)

name2performance = dict()
name2identified_gdf = dict()
name2intersection = dict()

# ==========================================
# CÓDIGO MODIFICADO: SOLO CONDADO DE RAMSEY
# ==========================================
# Comentamos el resto para hacer pruebas rápidas
# names = ['Carver_MN', 'Dakota_MN', 'Scott_MN', 'Hennepin_MN', 'Ramsey_MN', 'Washington_MN', 'Anoka_MN']
names = ['Ramsey_MN']

for name in names:
    gdf_official = name2gdf[name]
    
    # --- LÍNEA CORREGIDA ---
    gdf_official = gdf_official.to_crs(epsg=4326)
    # -----------------------
    
    polygons = gdf_official['geometry']

    multipolygon = polygons.union_all()
    multipolygon = multipolygon.convex_hull

    num_segments = 1
    gdf_identified, footprint_id2features, gdf_not_used = map_buildings.generate_gdf_with_segments(multipolygon, num_segments, map_buildings.tags)

    gdf_identified['geometry'] = gdf_identified['geometry'].apply(lambda geom: MultiPolygon([geom]) if isinstance(geom, Polygon) else geom)
    gdf_identified = gdf_identified[gdf_identified.geometry.type != 'Point']
    gdf_identified = gdf_identified[gdf_identified.geometry.type != 'LineString']    
    gdf_not_used = gdf_not_used[gdf_not_used.geometry.type == 'Point'] # just the points

    # Convert all to the utm_crs (use CRS for short distances)
    utm_crs = utils.get_utm_crs_from_geodataframe(gdf_official)
    gdf_official = gdf_official.to_crs(epsg=utm_crs)
    gdf_identified = gdf_identified.to_crs(epsg=utm_crs) 
    gdf_not_used =  gdf_not_used.to_crs(epsg=utm_crs)

    gdf_identified = map_buildings.use_auxiliary_data(gdf_identified, footprint_id2features)
    
    name2identified_gdf[name] = gdf_identified

    intersection = gdf_identified.copy()

    # Add column for expected results from groundtruth
    intersection_columns = intersection.progress_apply(lambda row: get_largest_overlap(row, gdf_official), axis=1)
    keep_lines = intersection_columns.notnull()
    intersection_columns = intersection_columns[keep_lines]
    intersection = intersection[keep_lines]
    intersection['org_type'] = intersection_columns.progress_apply(lambda row: row[0])
    intersection['old_label'] = intersection_columns.progress_apply(lambda row: row[1])
    intersection = intersection.dropna(subset=['org_type'])
    intersection.reset_index(inplace=True, drop=True)

    # =====================================================================
    # 🚀 MEJORAS AÑADIDAS: INTEGRACIÓN DEL LLM Y CORRECCIÓN POR ÁREA
    # =====================================================================
    print("\nAplicando clasificación mejorada (LLM + Análisis Geométrico)...")
    
    # 1. Cargar el diccionario LLM (Asegúrate de que el archivo existe en el mismo directorio)
    try:
        with open("diccionario_etiquetas_llm.json", "r", encoding="utf-8") as f:
            diccionario_llm = json.load(f)
    except FileNotFoundError:
        print("CUIDADO: No se encontró 'diccionario_etiquetas_llm.json'. Se usará la lógica base.")
        diccionario_llm = {}

    # 2. Guardamos la predicción original y calculamos el área (al estar en UTM, son metros cuadrados)
    intersection['type_original'] = intersection['type']
    intersection['area_m2'] = intersection.geometry.area

    # 3. Función inteligente de decisión
    def mejorar_clasificacion(row):
        tag = row['tag used']
        area = row['area_m2']
        tipo_original = row['type_original']

        # CASO A: El edificio NO tiene etiqueta en OSM
        if pd.isna(tag):
            if area > 800: # Umbral de 800m2 para considerarlo infraestructura industrial/comercial
                return 'NON_RES' 
            else:
                return 'RES' # Si es pequeño, asumimos casa (como decía el paper original)

        # CASO B: El edificio SÍ tiene etiqueta
        clasificacion_llm = diccionario_llm.get(tag, "N/A")

        if clasificacion_llm in ["RES", "NON_RES"]:
            return clasificacion_llm # Hacemos caso a nuestro diccionario mejorado
        else:
            # Si el LLM dijo "N/A" nos fiamos de lo que calculó tu modelo base original
            return tipo_original

    # Aplicamos la nueva lógica para sobreescribir la columna 'type'
    intersection['type'] = intersection.apply(mejorar_clasificacion, axis=1)
    # =====================================================================

    # Impresión de errores actualizada
    gdf_error = intersection[intersection['type'] != intersection['org_type']]
    print ("\nIncorrectly classified as NON_RES (Era RES pero fallamos):\n",Counter(gdf_error[gdf_error['type']=='NON_RES']['tag used']).most_common(10),"\n")
    print ("Incorrectly classified as RES (Era NON_RES pero fallamos):\n",Counter(gdf_error[gdf_error['type']=='RES']['tag used']).most_common(10),"\n")
    
    gdf_old_label = intersection[intersection['type'] != intersection['org_type']]
    print ("Incorrectly classified as NON_RES (old label):\n",Counter(gdf_old_label[gdf_old_label['type']=='NON_RES']['old_label']).most_common(10),"\n")
    print ("Incorrectly classified as RES (old label):\n",Counter(gdf_old_label[gdf_old_label['type']=='RES']['old_label']).most_common(10),"\n")

    name2intersection[name] = intersection

    official_values = intersection['org_type']
    predicted = intersection['type']
    performance = metrics.classification_report(official_values, predicted)

    name2performance[name] = performance
    print("=====================================================")
    print(f"RESULTADOS MEJORADOS PARA: {name}")
    print("=====================================================")
    print(performance)
    print("=====================================================\n\n")

path = os.path.join(out_path, 'comparison2')
utils.create_directory_if_not_exists(path)
with open(os.path.join(path,'name2performance.pickle'), 'wb') as handle:
    pickle.dump(name2performance, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(path,'name2identified_gdf.pickle'), 'wb') as handle:
    pickle.dump(name2identified_gdf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(path,'name2intersection.pickle'), 'wb') as handle:
    pickle.dump(name2intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)