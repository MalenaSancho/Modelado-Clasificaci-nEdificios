import os
import sys
import pickle
from collections import Counter

import pandas as pd
import geopandas as gpd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from pathlib import Path

# ============================================================
# CONFIGURACIÓN DE RUTAS DINÁMICAS
# ============================================================

# Detecta la localización del script actual (Alicante/scripts)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# Ajustamos la entrada a donde el script 00 guardó las variables de Alicante
input_path = BASE_DIR / "out" / "comparison_alicante"
output_path = BASE_DIR / "out" / "reporte_final"
output_path.mkdir(parents=True, exist_ok=True)

print(f"Cargando datos precalculados desde {input_path}...")

# OJO: Cargamos el archivo correcto con el sufijo '_alicante.pickle'
pickle_file = input_path / "name2intersection_alicante.pickle"

if not pickle_file.exists():
    raise FileNotFoundError(
        f"No se encuentra el archivo {pickle_file}. "
        f"Asegúrate de haber ejecutado el script de inspección/validación de Alicante primero."
    )

with open(pickle_file, "rb") as handle:
    name2intersection = pickle.load(handle)

# Inicializar PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# ============================================================
# PROCESAMIENTO Y GENERACIÓN DE REPORTE
# ============================================================

for name, intersection in name2intersection.items():
    print(f"Procesando región: {name}...")
    
    # Aseguramos que sea un GeoDataFrame para que no falle .to_file()
    if not isinstance(intersection, gpd.GeoDataFrame):
        intersection = gpd.GeoDataFrame(intersection, geometry="geometry")
    
    # --- EXPORTACIÓN DE DATOS ---
    # CSV: Quitamos la geometría para evitar conflictos de texto largo en Excel
    df_csv = pd.DataFrame(intersection.drop(columns=["geometry"], errors="ignore"))
    csv_out_path = output_path / f"{name}_resultados.csv"
    df_csv.to_csv(csv_out_path, index=False, encoding="utf-8")
    
    # GeoJSON: Exportación espacial de los polígonos con sus predicciones
    geojson_out_path = output_path / f"{name}_mapa.geojson"
    intersection.to_file(geojson_out_path, driver="GeoJSON")

    # --- CREACIÓN DEL PDF ---
    pdf.add_page()
    official_values = intersection["org_type"]
    predicted = intersection["type"]

    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, f"Resultados de Clasificacion: {name}", ln=True, align="C")
    pdf.ln(5)

    # 1. Métricas en texto plano
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "1. Metricas de Clasificacion", ln=True)
    pdf.set_font("Courier", size=10)
    reporte = metrics.classification_report(
        official_values, predicted, labels=["NON_RES", "RES"], zero_division=0
    )
    for linea in reporte.split("\n"):
        pdf.cell(0, 5, linea, ln=True)
    pdf.ln(5)

    # 2. Matriz de Confusión (Gráfico)
    plt.figure(figsize=(5, 3.5))
    cm = metrics.confusion_matrix(official_values, predicted, labels=["RES", "NON_RES"])
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["RES", "NON_RES"], 
        yticklabels=["RES", "NON_RES"]
    )
    plt.title(f"Matriz de Confusion - {name}")
    plt.tight_layout()
    
    imagen_filename = output_path / f"{name}_cm.png"
    plt.savefig(imagen_filename, dpi=300)
    plt.close()

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "2. Matriz de Confusion", ln=True)
    pdf.image(str(imagen_filename), x=45, w=120) 
    pdf.ln(5)

    # 3. Análisis de Errores Críticos (OSM Tags)
    gdf_error = intersection[intersection["type"] != intersection["org_type"]]
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "3. Top Errores de Clasificacion", ln=True)
    
    def agregar_errores_pdf(titulo, condicion):
        pdf.set_font("Helvetica", style="I", size=10)
        pdf.cell(0, 6, titulo, ln=True)
        pdf.set_font("Helvetica", size=10)
        
        # Filtramos errores bajo la condición dada
        errores_filtrados = gdf_error[condicion]
        if not errores_filtrados.empty and "tag used" in errores_filtrados.columns:
            errores = Counter(errores_filtrados["tag used"]).most_common(5)
            for tag, freq in errores:
                texto_tag = "Sin etiqueta en OSM (NaN)" if pd.isna(tag) else str(tag)
                pdf.cell(10)
                pdf.cell(0, 6, f"- {texto_tag}: {freq} veces", ln=True)
        else:
            pdf.cell(10)
            pdf.cell(0, 6, "- No se encontraron errores en esta categoría.", ln=True)
        pdf.ln(3)
        
    agregar_errores_pdf("a) Era NON_RES pero se clasifico como RES:", gdf_error["type"] == "RES")
    agregar_errores_pdf("b) Era RES pero se clasifico como NON_RES:", gdf_error["type"] == "NON_RES")

# Guardar documento final
archivo_pdf = output_path / "Reporte_Completo.pdf"
pdf.output(str(archivo_pdf))

print(f"\n¡Exito! CSVs, GeoJSONs y el Reporte PDF guardados en: {output_path}")