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
# CONFIGURACIÓN DE RUTAS DINÁMICAS (CORREGIDO)
# ============================================================

# Detecta automáticamente la localización de este script (Alicante/scripts)
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# Ajustado exactamente a las nuevas salidas del script de mejoras (out2 y comparison2_alicante)
input_path = BASE_DIR / "out2" / "comparison2_alicante"
output_path = BASE_DIR / "out2" / "reporte_final2"
output_path.mkdir(parents=True, exist_ok=True)

print(f"Cargando datos precalculados desde {input_path}...")

# Apuntamos al archivo real generado con sufijo '_alicante.pickle'
pickle_file = input_path / "name2intersection_alicante.pickle"

if not pickle_file.exists():
    raise FileNotFoundError(
        f"No se encuentra el archivo {pickle_file}. "
        f"Asegúrate de haber ejecutado primero el script de validación mejorada."
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
    print(f"Procesando: {name}...")
    
    # Forzar conversión a GeoDataFrame por seguridad antes de exportar a GeoJSON
    if not isinstance(intersection, gpd.GeoDataFrame):
        intersection = gpd.GeoDataFrame(intersection, geometry="geometry")
    
    # EXPORTACIÓN DE DATOS (CSV y GeoJSON)
    # CSV: Eliminamos la columna geometry de forma segura para no corromper el Excel/CSV
    df_csv = pd.DataFrame(intersection.drop(columns=['geometry'], errors='ignore'))
    df_csv.to_csv(output_path / f"{name}_resultados.csv", index=False, encoding="utf-8")
    
    # GeoJSON: Exportamos la geometría junto con las predicciones del LLM y área
    intersection.to_file(output_path / f"{name}_mapa.geojson", driver='GeoJSON')

    # --- CREACIÓN DEL PDF ---
    pdf.add_page()
    official_values = intersection['org_type']
    predicted = intersection['type']

    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, f"Resultados de Clasificacion Mejorada: {name}", ln=True, align='C')
    pdf.ln(5)

    # 1. Métricas de Clasificación
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "1. Metricas de Clasificacion", ln=True)
    pdf.set_font("Courier", size=10)
    
    # Aseguramos etiquetas estables y evitamos divisiones por cero si una clase es nula
    reporte = metrics.classification_report(
        official_values, 
        predicted, 
        labels=["NON_RES", "RES"], 
        zero_division=0
    )
    for linea in reporte.split('\n'):
        pdf.cell(0, 5, linea, ln=True)
    pdf.ln(5)

    # 2. Matriz de Confusión
    plt.figure(figsize=(5, 3.5))
    cm = metrics.confusion_matrix(official_values, predicted, labels=["RES", "NON_RES"])
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=["RES", "NON_RES"], 
        yticklabels=["RES", "NON_RES"]
    )
    plt.title(f'Matriz de Confusion (Mejorada) - {name}')
    plt.tight_layout()
    
    imagen_filename = output_path / f"{name}_cm.png"
    plt.savefig(imagen_filename, dpi=300)
    plt.close()

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "2. Matriz de Confusion", ln=True)
    # Pasamos a str() para evitar bugs de FPDF en Windows al leer objetos Path
    pdf.image(str(imagen_filename), x=45, w=120) 
    pdf.ln(5)

    # 3. Análisis de Errores Críticos
    gdf_error = intersection[intersection['type'] != intersection['org_type']]
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "3. Top Errores de Clasificacion", ln=True)
    
    def agregar_errores_pdf(titulo, condicion):
        pdf.set_font("Helvetica", style="I", size=10)
        pdf.cell(0, 6, titulo, ln=True)
        pdf.set_font("Helvetica", size=10)
        
        errores_filtrados = gdf_error[condicion]
        if not errores_filtrados.empty and 'tag used' in errores_filtrados.columns:
            errores = Counter(errores_filtrados['tag used']).most_common(5)
            for tag, freq in errores:
                texto_tag = "Sin etiqueta en OSM (NaN / Tratado por área)" if pd.isna(tag) else str(tag)
                pdf.cell(10)
                pdf.cell(0, 6, f"- {texto_tag}: {freq} veces", ln=True)
        else:
            pdf.cell(10)
            pdf.cell(0, 6, "- No se registraron errores bajo esta condicion.", ln=True)
        pdf.ln(3)

    agregar_errores_pdf("a) Era NON_RES pero se clasifico como RES:", gdf_error['type'] == 'RES')
    agregar_errores_pdf("b) Era RES pero se clasifico como NON_RES:", gdf_error['type'] == 'NON_RES')

# Guardar el documento final unificado en PDF
archivo_pdf = output_path / "Reporte_Completo.pdf"
pdf.output(str(archivo_pdf))

print(f"\n¡Exito! CSVs, GeoJSONs y el Reporte PDF guardados en: {output_path}")