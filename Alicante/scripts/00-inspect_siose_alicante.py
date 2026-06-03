# ============================================================
# SCRIPT: inspect_siose_alicante.py
# ============================================================
#
# OBJETIVO
# ------------------------------------------------------------
# Inspeccionar SIOSE dentro del municipio de Alicante/Alacant.
#
# Este script:
#   1. Lee los recintos municipales del IGN.
#   2. Filtra el municipio Alacant/Alicante.
#   3. Lee T_POLIGONOS.shp de SIOSE.
#   4. Recorta SIOSE al municipio de Alicante.
#   5. Lee T_VALORES.dbf y TC_SIOSE_COBERTURAS.dbf.
#   6. Une cada polígono con sus coberturas.
#   7. Obtiene la cobertura dominante por polígono.
#   8. Guarda CSV para que podamos definir RES / NON_RES / N/A.
#
# Salidas:
#   ../out/alicante_siose_inspection/siose_alicante_polygons.gpkg
#   ../out/alicante_siose_inspection/siose_alicante_dominant_coverages.csv
#   ../out/alicante_siose_inspection/siose_alicante_coverage_summary.csv
#
# ============================================================

from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# RUTAS
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

out_path = Path("../out/alicante_siose_inspection")
out_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# LEER MUNICIPIO DE ALICANTE
# ============================================================

print("\n=====================================================")
print("LEYENDO MUNICIPIOS")
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

# Nos quedamos con una única geometría municipal.
alicante = alicante.dissolve()


# ============================================================
# LEER SIOSE POLÍGONOS
# ============================================================

print("\n=====================================================")
print("LEYENDO T_POLIGONOS")
print("=====================================================")

siose = gpd.read_file(siose_poligonos_path)

print("CRS SIOSE:", siose.crs)
print("Polígonos SIOSE provincia:", siose.shape[0])

# Reproyectamos Alicante al CRS de SIOSE.
alicante = alicante.to_crs(siose.crs)


# ============================================================
# RECORTE ESPACIAL
# ============================================================
#
# Primero hacemos filtro rápido por bounding box/intersección.
# Después hacemos overlay para recortar exactamente al municipio.
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

siose_alicante = siose_alicante.dropna(subset=["geometry"])
siose_alicante = siose_alicante[
    siose_alicante.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()

siose_alicante["clip_area_m2"] = siose_alicante.geometry.area
siose_alicante["clip_area_ha"] = siose_alicante["clip_area_m2"] / 10000

print("Polígonos SIOSE recortados:", siose_alicante.shape[0])

siose_alicante_gpkg = out_path / "siose_alicante_polygons.gpkg"
siose_alicante.to_file(siose_alicante_gpkg, driver="GPKG")

print("Guardado:")
print(siose_alicante_gpkg)


# ============================================================
# LEER TABLAS AUXILIARES
# ============================================================

print("\n=====================================================")
print("LEYENDO TABLAS AUXILIARES")
print("=====================================================")

valores = gpd.read_file(t_valores_path)
coberturas = gpd.read_file(tc_coberturas_path)

# Las tablas DBF se leen como GeoDataFrame sin geometry real.
# Las convertimos a DataFrame normal.
valores = pd.DataFrame(valores)
coberturas = pd.DataFrame(coberturas)

print("T_VALORES:", valores.shape)
print("TC_SIOSE_COBERTURAS:", coberturas.shape)

print("\nColumnas T_VALORES:")
print(list(valores.columns))

print("\nColumnas TC_SIOSE_COBERTURAS:")
print(list(coberturas.columns))


# ============================================================
# FILTRAR T_VALORES A LOS POLÍGONOS DE ALICANTE
# ============================================================

ids_alicante = set(siose_alicante["ID_POLYGON"].astype(str))

valores_alicante = valores[
    valores["ID_POLYGON"].astype(str).isin(ids_alicante)
].copy()

print("\nFilas de T_VALORES asociadas a Alicante:", valores_alicante.shape[0])


# ============================================================
# UNIR VALORES CON DESCRIPCIÓN DE COBERTURA
# ============================================================

valores_alicante["ID_COBER"] = valores_alicante["ID_COBER"].astype(int)
coberturas["ID_COBER"] = coberturas["ID_COBER"].astype(int)

valores_desc = valores_alicante.merge(
    coberturas[
        [
            "ID_COBER",
            "CODE_ABREV",
            "DESC_COBER",
            "COB_PADRES",
            "LIST_ATRIB",
            "LIST_OBLIG",
            "LIST_OPCIO",
        ]
    ],
    on="ID_COBER",
    how="left"
)

# Guardamos todos los componentes.
all_components_path = out_path / "siose_alicante_all_components.csv"
valores_desc.to_csv(all_components_path, index=False, encoding="utf-8")

print("Guardado componentes completos:")
print(all_components_path)


# ============================================================
# COBERTURA DOMINANTE POR POLÍGONO
# ============================================================
#
# Cada polígono puede tener varias coberturas con distintos porcentajes.
# Tomamos como dominante la cobertura con mayor SUPERF_POR.
# ============================================================

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

# Añadimos SIOSE_CODE original.
dominant = dominant.merge(
    siose_alicante[
        [
            "ID_POLYGON",
            "SIOSE_CODE",
            "SUPERF_HA",
            "SUP_HA",
            "clip_area_ha",
        ]
    ],
    on="ID_POLYGON",
    how="left"
)

dominant_path = out_path / "siose_alicante_dominant_coverages.csv"
dominant.to_csv(dominant_path, index=False, encoding="utf-8")

print("Guardado coberturas dominantes:")
print(dominant_path)


# ============================================================
# RESUMEN POR COBERTURA DOMINANTE
# ============================================================

summary = (
    dominant
    .groupby(["ID_COBER", "CODE_ABREV", "DESC_COBER"], dropna=False)
    .agg(
        n_polygons=("ID_POLYGON", "count"),
        total_clip_area_ha=("clip_area_ha", "sum"),
        mean_dominant_percentage=("SUPERF_POR", "mean"),
    )
    .reset_index()
    .sort_values("total_clip_area_ha", ascending=False)
)

summary_path = out_path / "siose_alicante_coverage_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8")

print("Guardado resumen de coberturas:")
print(summary_path)


# ============================================================
# MOSTRAR RESUMEN EN TERMINAL
# ============================================================

print("\n=====================================================")
print("RESUMEN DE COBERTURAS DOMINANTES EN ALICANTE")
print("=====================================================")

print(summary.head(80).to_string(index=False))

print("\n=====================================================")
print("PROCESO TERMINADO")
print("=====================================================")
print("Archivos generados:")
print(siose_alicante_gpkg)
print(all_components_path)
print(dominant_path)
print(summary_path)