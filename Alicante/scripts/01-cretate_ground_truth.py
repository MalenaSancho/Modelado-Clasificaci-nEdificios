# ============================================================
# SCRIPT: create_ground_truth_alicante.py
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
# Crear el dataset de validación para la ciudad de Alicante
# usando SIOSE 2015/2014 de la Comunitat Valenciana.
#
# Este script genera un archivo equivalente al que antes creamos
# para Minneapolis/St. Paul con GeneralizedLandUse2020:
#
#     ../data/test_region2gdf_alicante.pickle
#
# La salida será un diccionario:
#
#     name2gdf["Alicante"] = gdf_alicante
#
# donde gdf_alicante contiene:
#
#     geometry   -> geometría SIOSE recortada al municipio
#     old_label  -> etiqueta original SIOSE
#     org_type   -> RES / NON_RES
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import sys
import pickle
from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

homedir = "../"
codedir = os.path.join(homedir, "src")
sys.path.append(codedir)

from utils import create_directory_if_not_exists


# ============================================================
# RUTAS DE ENTRADA
# ============================================================

municipios_path = Path(
    "../raw_data/limites_municipales/"
    "recintos_municipales_inspire_peninbal_etrs89/"
    "recintos_municipales_inspire_peninbal_etrs89.shp"
)

siose_poligonos_path = Path(
    "../raw_data/siose_cv/T_POLIGONOS.shp"
)

t_valores_path = Path(
    "../raw_data/siose_cv/T_VALORES.dbf"
)

tc_coberturas_path = Path(
    "../raw_data/siose_cv/TC_SIOSE_COBERTURAS.dbf"
)


# ============================================================
# RUTAS DE SALIDA
# ============================================================

data_path = Path("../data")
out_path = Path("../out/alicante_ground_truth")

create_directory_if_not_exists(data_path)
create_directory_if_not_exists(out_path)

pickle_out_path = data_path / "test_region2gdf_alicante.pickle"
gpkg_out_path = out_path / "alicante_ground_truth.gpkg"
csv_summary_out_path = out_path / "alicante_ground_truth_summary.csv"
csv_all_out_path = out_path / "alicante_ground_truth_all_polygons.csv"


# ============================================================
# DICCIONARIOS DE RECLASIFICACIÓN
# ============================================================
#
# A partir del resumen que hemos obtenido para Alicante:
#
#   UCS  Casco          -> RES
#   UEN  Ensanche       -> RES
#   UDS  Discontinuo    -> RES
#
#   IPO  Polígono Industrial Ordenado      -> NON_RES
#   IPS  Polígono Industrial sin Ordenar   -> NON_RES
#   IAS  Industrial Aislada                -> NON_RES
#   TCO  Comercial y Oficinas              -> NON_RES
#   TCH  Complejo Hotelero                 -> NON_RES
#   EDU  Educación                         -> NON_RES
#   ESN  Sanitario                         -> NON_RES
#   EAI  Administrativo Institucional      -> NON_RES
#   etc.
#
# El resto de clases agrícolas, naturales, agua, ramblas, carreteras,
# vías ferroviarias, parques, playas, matorral, pastizal, etc. se
# eliminan como N/A porque no son una verdad clara de edificios
# residenciales/no residenciales.
#
# ============================================================

RES_CODES = {
    "UCS",  # Casco
    "UEN",  # Ensanche
    "UDS",  # Discontinuo
}

NON_RES_CODES = {
    "IPO",  # Polígono Industrial Ordenado
    "IPS",  # Polígono Industrial sin Ordenar
    "IAS",  # Industrial Aislada
    "TCO",  # Comercial y Oficinas
    "TCH",  # Complejo Hotelero
    "EAI",  # Administrativo Institucional
    "ESN",  # Sanitario
    "ECM",  # Cementerio
    "EDU",  # Educación
    "EPN",  # Penitenciario
    "ECL",  # Cultural
    "EDP",  # Deportivo
    "NPO",  # Portuario
    "NEL",  # Eléctrica
    "NDP",  # Depuradoras y Potabilizadoras
    "NCC",  # Conducciones y Canales
    "PMX",  # Minero Extractivo
    "ZEV",  # Zonas de Extracción o Vertido
    "PAG",  # Agrícola, Ganadero
    "OCT",  # Otras Construcciones
}

NA_CODES = {
    "PST",  # Pastizal
    "LFN",  # Frutales No Cítricos
    "LFC",  # Frutales Cítricos
    "SDN",  # Suelo Desnudo
    "CHL",  # Cultivos Herbáceos
    "ECG",  # Campo de Golf
    "MTR",  # Matorral
    "LVI",  # Viñedo
    "NRV",  # Red Viaria
    "NRF",  # Red Ferroviaria
    "EPU",  # Parque Urbano
    "HSM",  # Salinas Marinas
    "PDA",  # Playas, dunas y arenales
    "LAA",  # Lámina de Agua Artificial
    "HMA",  # Marismas
    "LOL",  # Olivar
    "RMB",  # Ramblas
    "SNE",  # Suelo No Edificado
    "CNF",  # Coníferas
    "LOC",  # Otros Leñosos
    "ALG",  # Lagos y Lagunas
    "ARR",  # Afloramientos Rocosos
    "VAP",  # Vial, aparcamiento o zona peatonal sin vegetación
    "ACM",  # Acantilados Marinos
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def save_pickle(name2gdf, path):
    """
    Guarda el diccionario name2gdf en formato pickle.
    """
    with open(path, "wb") as handle:
        pickle.dump(name2gdf, handle, protocol=pickle.HIGHEST_PROTOCOL)


def clean_geometries(gdf):
    """
    Limpia geometrías vacías, nulas o no poligonales.
    """
    gdf = gdf.copy()

    gdf = gdf.dropna(subset=["geometry"])
    gdf = gdf[gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    gdf.reset_index(drop=True, inplace=True)

    return gdf


def dominant_coverage_by_polygon(valores, coberturas, siose_alicante):
    """
    Obtiene la cobertura dominante de cada polígono SIOSE.

    Cada polígono puede estar formado por varias coberturas con distintos
    porcentajes. Tomamos como dominante la cobertura con mayor SUPERF_POR.
    """
    valores = pd.DataFrame(valores).copy()
    coberturas = pd.DataFrame(coberturas).copy()

    ids_alicante = set(siose_alicante["ID_POLYGON"].astype(str))

    valores_alicante = valores[
        valores["ID_POLYGON"].astype(str).isin(ids_alicante)
    ].copy()

    valores_alicante["ID_COBER"] = valores_alicante["ID_COBER"].astype(int)
    coberturas["ID_COBER"] = coberturas["ID_COBER"].astype(int)

    valores_desc = valores_alicante.merge(
        coberturas[
            [
                "ID_COBER",
                "CODE_ABREV",
                "DESC_COBER",
            ]
        ],
        on="ID_COBER",
        how="left"
    )

    valores_desc["SUPERF_POR"] = pd.to_numeric(
        valores_desc["SUPERF_POR"],
        errors="coerce"
    )

    dominant = (
        valores_desc
        .sort_values(["ID_POLYGON", "SUPERF_POR"], ascending=[True, False])
        .drop_duplicates(subset=["ID_POLYGON"], keep="first")
        .copy()
    )

    dominant = dominant[
        [
            "ID_POLYGON",
            "ID_COBER",
            "CODE_ABREV",
            "DESC_COBER",
            "SUPERF_POR",
            "ATRIBUTOS",
        ]
    ].copy()

    return dominant


def classify_by_code_abrev(code):
    """
    Clasifica una cobertura dominante SIOSE por su CODE_ABREV.
    """
    if pd.isna(code):
        return "N/A"

    code = str(code).strip()

    if code in RES_CODES:
        return "RES"

    if code in NON_RES_CODES:
        return "NON_RES"

    if code in NA_CODES:
        return "N/A"

    return "N/A"


def extract_siose_codes(siose_code):
    """
    Extrae códigos SIOSE de 3 letras desde SIOSE_CODE.

    Ejemplos:
        UEN(70EDFem_20ZAU_10VAP) -> UEN, EDF, ZAU, VAP
        I(95PST_05EDFea)         -> PST, EDF
        R(70PSTpc_30IAS(...))    -> PST, IAS
    """
    if pd.isna(siose_code):
        return []

    text = str(siose_code)

    codes = []

    # Extraemos secuencias de 3 letras mayúsculas.
    # Esto captura códigos como UEN, UDS, IAS, TCO, PST, CNF, etc.
    import re
    matches = re.findall(r"[A-Z]{3}", text)

    for m in matches:
        codes.append(m)

    return codes


def classify_by_siose_code(siose_code):
    """
    Clasifica usando los códigos que aparecen dentro de SIOSE_CODE.

    Esto es importante para los casos ID_COBER = 600 / No Predefinida,
    donde la cobertura dominante no es directamente informativa, pero
    SIOSE_CODE sí contiene información compuesta.

    Estrategia:
      - Si aparece algún código residencial urbano, clasificamos RES.
      - Si no aparece residencial, pero aparece código no residencial,
        clasificamos NON_RES.
      - Si solo aparecen códigos naturales, agrícolas, agua, carreteras,
        etc., clasificamos N/A.
    """
    codes = extract_siose_codes(siose_code)

    if len(codes) == 0:
        return "N/A"

    if any(code in RES_CODES for code in codes):
        return "RES"

    if any(code in NON_RES_CODES for code in codes):
        return "NON_RES"

    return "N/A"


def classify_polygon(row):
    """
    Clasifica un polígono SIOSE como RES, NON_RES o N/A.

    Criterio corregido:
    - Si la cobertura dominante es claramente RES, usamos RES.
    - Si la cobertura dominante es claramente NON_RES, usamos NON_RES.
    - Si la cobertura dominante es claramente N/A, respetamos N/A.
      Esto evita que una carretera NRV pase a NON_RES solo porque dentro
      de SIOSE_CODE aparezca una pequeña componente OCT.
    - Solo usamos SIOSE_CODE como apoyo cuando la cobertura dominante es
      No Predefinida o no tiene CODE_ABREV útil.
    """
    code_abrev = row.get("CODE_ABREV", None)
    siose_code = row.get("SIOSE_CODE", None)

    if pd.notna(code_abrev):
        code_abrev = str(code_abrev).strip()

        if code_abrev in RES_CODES:
            return "RES"

        if code_abrev in NON_RES_CODES:
            return "NON_RES"

        if code_abrev in NA_CODES:
            return "N/A"

    # Solo si CODE_ABREV no informa bien, usamos SIOSE_CODE.
    return classify_by_siose_code(siose_code)


# ============================================================
# COMPROBACIÓN DE ARCHIVOS
# ============================================================

required_paths = [
    municipios_path,
    siose_poligonos_path,
    t_valores_path,
    tc_coberturas_path,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


# ============================================================
# LEER MUNICIPIOS Y FILTRAR ALICANTE
# ============================================================

print("\n=====================================================")
print("LEYENDO LÍMITES MUNICIPALES")
print("=====================================================")

municipios = gpd.read_file(municipios_path)

alicante = municipios[
    municipios["NAMEUNIT"].astype(str).str.contains(
        "Alacant/Alicante",
        case=False,
        na=False
    )
].copy()

if alicante.empty:
    raise ValueError("No se ha encontrado el municipio Alacant/Alicante.")

print("Municipio encontrado:")
print(alicante[["NAMEUNIT", "NATCODE"]])

alicante = alicante.dissolve()


# ============================================================
# LEER SIOSE
# ============================================================

print("\n=====================================================")
print("LEYENDO SIOSE")
print("=====================================================")

siose = gpd.read_file(siose_poligonos_path)

print("CRS SIOSE:", siose.crs)
print("Polígonos SIOSE provincia:", siose.shape[0])

alicante = alicante.to_crs(siose.crs)


# ============================================================
# RECORTAR SIOSE AL MUNICIPIO
# ============================================================

print("\n=====================================================")
print("RECORTANDO SIOSE AL MUNICIPIO DE ALICANTE")
print("=====================================================")

alicante_geom = alicante.geometry.iloc[0]

siose_bbox = siose[siose.intersects(alicante_geom)].copy()

print("Polígonos que intersectan Alicante:", siose_bbox.shape[0])

siose_alicante = gpd.overlay(
    siose_bbox,
    alicante[["geometry"]],
    how="intersection"
)

siose_alicante = clean_geometries(siose_alicante)

siose_alicante["clip_area_m2"] = siose_alicante.geometry.area
siose_alicante["clip_area_ha"] = siose_alicante["clip_area_m2"] / 10000

print("Polígonos SIOSE recortados:", siose_alicante.shape[0])


# ============================================================
# LEER TABLAS AUXILIARES Y OBTENER COBERTURA DOMINANTE
# ============================================================

print("\n=====================================================")
print("LEYENDO TABLAS AUXILIARES SIOSE")
print("=====================================================")

valores = gpd.read_file(t_valores_path)
coberturas = gpd.read_file(tc_coberturas_path)

dominant = dominant_coverage_by_polygon(
    valores=valores,
    coberturas=coberturas,
    siose_alicante=siose_alicante
)

print("Coberturas dominantes obtenidas:", dominant.shape[0])


# ============================================================
# UNIR COBERTURA DOMINANTE CON POLÍGONOS
# ============================================================

gdf = siose_alicante.merge(
    dominant,
    on="ID_POLYGON",
    how="left"
)

print("Polígonos tras unión con cobertura dominante:", gdf.shape[0])


# ============================================================
# CLASIFICAR RES / NON_RES / N/A
# ============================================================

print("\n=====================================================")
print("CLASIFICANDO POLÍGONOS SIOSE")
print("=====================================================")

gdf["org_type"] = gdf.apply(classify_polygon, axis=1)

gdf["old_label"] = (
    "ID_COBER="
    + gdf["ID_COBER"].fillna("missing").astype(str)
    + " | CODE_ABREV="
    + gdf["CODE_ABREV"].fillna("missing").astype(str)
    + " | DESC_COBER="
    + gdf["DESC_COBER"].fillna("missing").astype(str)
    + " | SIOSE_CODE="
    + gdf["SIOSE_CODE"].fillna("missing").astype(str)
)

print("\nDistribución antes de eliminar N/A:")
print(gdf["org_type"].value_counts(dropna=False))

# Guardamos todos los polígonos clasificados, incluidos N/A, para revisar.
gdf.drop(columns="geometry").to_csv(
    csv_all_out_path,
    index=False,
    encoding="utf-8"
)

# Eliminamos N/A para quedarnos con verdad binaria.
gdf_binary = gdf[gdf["org_type"].isin(["RES", "NON_RES"])].copy()
gdf_binary.reset_index(drop=True, inplace=True)

print("\nDistribución final RES / NON_RES:")
print(gdf_binary["org_type"].value_counts(dropna=False))

summary = (
    gdf
    .groupby(["org_type", "CODE_ABREV", "DESC_COBER"], dropna=False)
    .agg(
        n_polygons=("ID_POLYGON", "count"),
        total_area_ha=("clip_area_ha", "sum"),
    )
    .reset_index()
    .sort_values(["org_type", "total_area_ha"], ascending=[True, False])
)

summary.to_csv(csv_summary_out_path, index=False, encoding="utf-8")


# ============================================================
# PREPARAR GDF FINAL
# ============================================================

gdf_final = gdf_binary[
    [
        "geometry",
        "org_type",
        "old_label",
        "ID_POLYGON",
        "SIOSE_CODE",
        "ID_COBER",
        "CODE_ABREV",
        "DESC_COBER",
        "clip_area_ha",
    ]
].copy()

gdf_final = clean_geometries(gdf_final)

# Guardar GPKG para inspección visual en QGIS.
gdf_final.to_file(gpkg_out_path, driver="GPKG")


# ============================================================
# GUARDAR PICKLE
# ============================================================

name2gdf = {
    "Alicante": gdf_final
}

save_pickle(name2gdf, pickle_out_path)


# ============================================================
# MENSAJE FINAL
# ============================================================

print("\n=====================================================")
print("GROUND TRUTH DE ALICANTE CREADO CORRECTAMENTE")
print("=====================================================")

print("\nArchivo pickle generado:")
print(pickle_out_path)

print("\nArchivo GPKG generado:")
print(gpkg_out_path)

print("\nResumen generado:")
print(csv_summary_out_path)

print("\nDistribución final:")
print(gdf_final["org_type"].value_counts(dropna=False))

print("\nPrimeras filas:")
print(gdf_final[["org_type", "old_label"]].head(10).to_string(index=False))

print("=====================================================")