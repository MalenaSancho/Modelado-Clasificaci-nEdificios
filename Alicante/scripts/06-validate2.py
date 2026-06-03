import os
import sys
import pickle
from collections import Counter
import json

import pandas as pd
import osmnx as ox
from sklearn import metrics
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

import map_buildings
import utils

tqdm.pandas()


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

homedir = "../"
codedir = os.path.join(homedir, "src")
sys.path.append(codedir)

# ============================================================
# RUTAS DE ENTRADA Y SALIDA
# ============================================================

in_path = "../data/"
out_path = "../out2/"
cache_folder = "../osmnx_cache2/"


# ============================================================
# CONFIGURACIÓN DE OSMNX
# ============================================================

ox.settings.cache_folder = cache_folder
ox.settings.use_cache = True
ox.settings.requests_timeout = 500

# Si Overpass falla mucho, puedes activar este servidor alternativo:
# ox.settings.overpass_url = "https://overpass.kumi.systems/api"


# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def get_largest_overlap(row, gdf_official):
    """
    Para cada edificio clasificado por el modelo, buscamos el polígono
    del ground truth SIOSE con el que más se solapa.

    Devuelve:
        - org_type: clase oficial RES / NON_RES
        - old_label: etiqueta original SIOSE
    """
    overlaps = gdf_official[gdf_official.intersects(row.geometry)]

    if overlaps.empty:
        return None

    largest_overlap_index = overlaps.apply(
        lambda x: x.geometry.intersection(row.geometry).area,
        axis=1
    ).idxmax()

    return (
        gdf_official.loc[largest_overlap_index]["org_type"],
        gdf_official.loc[largest_overlap_index]["old_label"]
    )


# ============================================================
# CARGA DEL GROUND TRUTH DE ALICANTE
# ============================================================

with open(os.path.join(in_path, "test_region2gdf_alicante.pickle"), "rb") as handle:
    name2gdf = pickle.load(handle)


# ============================================================
# DICCIONARIOS DE RESULTADOS
# ============================================================

name2performance = dict()
name2identified_gdf = dict()
name2intersection = dict()


# ============================================================
# REGIONES A VALIDAR
# ============================================================

names = ["Alicante"]


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

for name in names:

    print("=====================================================")
    print(f"VALIDANDO {name}")
    print("=====================================================")

    # ------------------------------------------------------------
    # 1. Cargar ground truth oficial SIOSE
    # ------------------------------------------------------------

    if name not in name2gdf:
        raise KeyError(
            f"No existe la clave '{name}' en test_region2gdf_alicante.pickle. "
            f"Claves disponibles: {list(name2gdf.keys())}"
        )

    gdf_official = name2gdf[name].copy()

    print("\nDistribución oficial del ground truth:")
    print(gdf_official["org_type"].value_counts(dropna=False))

    # OSMnx necesita geometrías en latitud/longitud.
    gdf_official = gdf_official.to_crs(epsg=4326)

    polygons = gdf_official["geometry"]

    # ------------------------------------------------------------
    # 2. Crear área de descarga
    # ------------------------------------------------------------
    #
    # Usamos la unión de los polígonos SIOSE RES/NON_RES y luego
    # el convex_hull para obtener un polígono único de descarga.
    #
    # ------------------------------------------------------------

    multipolygon = polygons.union_all()
    multipolygon = multipolygon.convex_hull

    # ------------------------------------------------------------
    # 3. Descargar edificios OSM y clasificarlos
    # ------------------------------------------------------------
    #
    # Para Alicante empezamos con 3 segmentos.
    # Si Overpass da timeout, sube a 5.
    #
    # ------------------------------------------------------------

    num_segments = 3

    print("\nDescargando edificios OSM...")
    print(f"Número de segmentos por lado: {num_segments}")
    print(f"Total aproximado de segmentos: {num_segments * num_segments}")

    gdf_identified, footprint_id2features, gdf_not_used = map_buildings.generate_gdf_with_segments(
        multipolygon,
        num_segments,
        map_buildings.tags
    )

    print("\nEdificios descargados/clasificados inicialmente:")
    print(gdf_identified.shape)

    if gdf_identified.shape[0] == 0:
        print(f"No se han descargado edificios para {name}.")
        continue

    # ------------------------------------------------------------
    # 4. Normalizar geometrías
    # ------------------------------------------------------------

    gdf_identified["geometry"] = gdf_identified["geometry"].apply(
        lambda geom: MultiPolygon([geom]) if isinstance(geom, Polygon) else geom
    )

    gdf_identified = gdf_identified[gdf_identified.geometry.type != "Point"]
    gdf_identified = gdf_identified[gdf_identified.geometry.type != "LineString"]

    if gdf_not_used is not None and gdf_not_used.shape[0] > 0:
        gdf_not_used = gdf_not_used[gdf_not_used.geometry.type == "Point"]

    # ------------------------------------------------------------
    # 5. Convertir todo a CRS UTM
    # ------------------------------------------------------------
    #
    # Para calcular áreas de solapamiento necesitamos trabajar en un CRS
    # proyectado, no en latitud/longitud.
    #
    # ------------------------------------------------------------

    utm_crs = utils.get_utm_crs_from_geodataframe(gdf_official)

    gdf_official = gdf_official.to_crs(epsg=utm_crs)
    gdf_identified = gdf_identified.to_crs(epsg=utm_crs)

    if gdf_not_used is not None and gdf_not_used.shape[0] > 0:
        gdf_not_used = gdf_not_used.to_crs(epsg=utm_crs)

    # ------------------------------------------------------------
    # 6. Aplicar información auxiliar
    # ------------------------------------------------------------

    gdf_identified = map_buildings.use_auxiliary_data(
        gdf_identified,
        footprint_id2features
    )

    print("\nDistribución de clases predichas antes de mejora:")
    print(gdf_identified["type"].value_counts(dropna=False))

    if "aux info" in gdf_identified.columns:
        print("\nDistribución de aux info:")
        print(gdf_identified["aux info"].value_counts(dropna=False))

    name2identified_gdf[name] = gdf_identified

    # ------------------------------------------------------------
    # 7. Cruzar edificios clasificados con ground truth
    # ------------------------------------------------------------

    intersection = gdf_identified.copy()

    print("\nBuscando mayor solapamiento con el ground truth...")

    intersection_columns = intersection.progress_apply(
        lambda row: get_largest_overlap(row, gdf_official),
        axis=1
    )

    keep_lines = intersection_columns.notnull()

    intersection_columns = intersection_columns[keep_lines]
    intersection = intersection[keep_lines].copy()

    intersection["org_type"] = intersection_columns.progress_apply(
        lambda row: row[0]
    )

    intersection["old_label"] = intersection_columns.progress_apply(
        lambda row: row[1]
    )

    intersection = intersection.dropna(subset=["org_type"])
    intersection.reset_index(inplace=True, drop=True)

    print("\nNúmero de edificios con solapamiento válido:")
    print(intersection.shape[0])

    if intersection.shape[0] == 0:
        print(f"No hay edificios validables para {name}.")
        continue

    # =====================================================================
    # MEJORAS AÑADIDAS: INTEGRACIÓN DEL LLM Y CORRECCIÓN POR ÁREA
    # =====================================================================

    print("\nAplicando clasificación mejorada para Alicante (LLM + análisis geométrico)...")

    # ------------------------------------------------------------
    # 1. Cargar diccionario LLM
    # ------------------------------------------------------------
    #
    # El archivo debe estar en la carpeta desde la que ejecutas el script.
    # Si ejecutas desde scripts/, ponlo dentro de scripts/.
    #
    # Ejemplo:
    #     scripts/diccionario_etiquetas_llm.json
    #
    # Formato esperado:
    #     {
    #       "building:house": "RES",
    #       "building:apartments": "RES",
    #       "building:commercial": "NON_RES",
    #       "amenity:school": "NON_RES",
    #       ...
    #     }
    #
    # ------------------------------------------------------------

    try:
        with open("diccionario_etiquetas_llm_alicante.json", "r", encoding="utf-8") as f:
            diccionario_llm = json.load(f)
        print("Diccionario LLM cargado correctamente.")
        print(f"Número de etiquetas en diccionario: {len(diccionario_llm)}")
    except FileNotFoundError:
        print("CUIDADO: No se encontró 'diccionario_etiquetas_llm_alicante.json'.")
        print("Se usará únicamente la lógica base + corrección por área para edificios sin etiqueta.")
        diccionario_llm = {}

    # ------------------------------------------------------------
    # 2. Guardar predicción original y calcular área
    # ------------------------------------------------------------
    #
    # Como ya estamos en UTM, geometry.area está en m².
    #
    # ------------------------------------------------------------

    intersection["type_original"] = intersection["type"]
    intersection["area_m2"] = intersection.geometry.area

    # ------------------------------------------------------------
    # 3. Función de mejora de clasificación
    # ------------------------------------------------------------

    def mejorar_clasificacion(row):
        """
        Mejora la clasificación original usando:
        1. Diccionario LLM de etiquetas.
        2. Corrección geométrica por área en edificios sin etiqueta.

        Criterio:
        - Si no hay tag:
              área > 800 m²  -> NON_RES
              área <= 800 m² -> RES
        - Si hay tag y el diccionario LLM conoce la etiqueta:
              usamos RES / NON_RES del diccionario.
        - Si el diccionario no conoce la etiqueta o devuelve N/A:
              mantenemos el tipo original del modelo base.
        """
        tag = row["tag used"]
        area = row["area_m2"]
        tipo_original = row["type_original"]

        # CASO A: edificio sin etiqueta OSM clara.
        if pd.isna(tag) or tag in [None, "None", "nan", "NaN", ""]:
            if area > 800:
                return "NON_RES"
            else:
                return "RES"

        # CASO B: edificio con etiqueta OSM.
        tag = str(tag)

        clasificacion_llm = diccionario_llm.get(tag, "N/A")

        if clasificacion_llm in ["RES", "NON_RES"]:
            return clasificacion_llm

        # Si el LLM no aporta información útil, mantenemos el modelo base.
        return tipo_original

    # Aplicamos la nueva lógica para sobreescribir la columna type.
    intersection["type"] = intersection.apply(mejorar_clasificacion, axis=1)

    print("\nDistribución de clases predichas después de mejora:")
    print(intersection["type"].value_counts(dropna=False))

    # =====================================================================
    # FIN DE MEJORAS
    # =====================================================================

    # ------------------------------------------------------------
    # 8. Analizar errores
    # ------------------------------------------------------------

    gdf_error = intersection[intersection["type"] != intersection["org_type"]]

    print("\nIncorrectly classified as NON_RES (era RES pero predijimos NON_RES):")
    print(Counter(gdf_error[gdf_error["type"] == "NON_RES"]["tag used"]).most_common(10))
    print()

    print("Incorrectly classified as RES (era NON_RES pero predijimos RES):")
    print(Counter(gdf_error[gdf_error["type"] == "RES"]["tag used"]).most_common(10))
    print()

    print("Incorrectly classified as NON_RES (old label):")
    print(Counter(gdf_error[gdf_error["type"] == "NON_RES"]["old_label"]).most_common(10))
    print()

    print("Incorrectly classified as RES (old label):")
    print(Counter(gdf_error[gdf_error["type"] == "RES"]["old_label"]).most_common(10))
    print()

    name2intersection[name] = intersection

    # ------------------------------------------------------------
    # 9. Calcular métricas
    # ------------------------------------------------------------

    official_values = intersection["org_type"]
    predicted = intersection["type"]

    performance = metrics.classification_report(
        official_values,
        predicted,
        labels=["NON_RES", "RES"],
        zero_division=0
    )

    name2performance[name] = performance

    print("=====================================================")
    print(f"RESULTADOS MEJORADOS PARA: {name}")
    print("=====================================================")
    print(performance)
    print("=====================================================\n\n")


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

path = os.path.join(out_path, "comparison2_alicante")
utils.create_directory_if_not_exists(out_path)
utils.create_directory_if_not_exists(path)

with open(os.path.join(path, "name2performance_alicante.pickle"), "wb") as handle:
    pickle.dump(name2performance, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(path, "name2identified_gdf_alicante.pickle"), "wb") as handle:
    pickle.dump(name2identified_gdf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(path, "name2intersection_alicante.pickle"), "wb") as handle:
    pickle.dump(name2intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n=====================================================")
print("VALIDACIÓN MEJORADA DE ALICANTE TERMINADA")
print("=====================================================")
print("Resultados guardados en:")
print(path)