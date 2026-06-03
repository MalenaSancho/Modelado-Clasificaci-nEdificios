import pickle
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from fpdf import FPDF

# Rutas
input_path = "../out/comparison" # Ajustado a tu ruta actual
output_path = "../out/reporte_final"

if not os.path.exists(output_path):
    os.makedirs(output_path)

print(f"Cargando datos precalculados desde {input_path}...")
with open(os.path.join(input_path, 'name2intersection.pickle'), 'rb') as handle:
    name2intersection = pickle.load(handle)

# Inicializar PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

for name, intersection in name2intersection.items():
    print(f"Procesando: {name}...")
    
    # EXPORTACIÓN DE DATOS (CSV y GeoJSON)
    # CSV: Quitamos la geometría porque Excel no entiende polígonos
    df_csv = pd.DataFrame(intersection.drop(columns=['geometry']))
    df_csv.to_csv(os.path.join(output_path, f"{name}_resultados.csv"), index=False)
    
    # GeoJSON: Exportamos directamente 'intersection', que contiene los polígonos y la predicción MEJORADA
    intersection.to_file(os.path.join(output_path, f"{name}_mapa.geojson"), driver='GeoJSON')

    # --- CREACIÓN DEL PDF ---
    pdf.add_page()
    official_values = intersection['org_type']
    predicted = intersection['type']

    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, f"Resultados de Clasificacion: {name}", ln=True, align='C')
    pdf.ln(5)

    # 1. Métricas
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "1. Metricas de Clasificacion", ln=True)
    pdf.set_font("Courier", size=10)
    reporte = metrics.classification_report(official_values, predicted)
    for linea in reporte.split('\n'):
        pdf.cell(0, 5, linea, ln=True)
    pdf.ln(5)

    # 2. Matriz de Confusión
    plt.figure(figsize=(5, 3.5))
    cm = metrics.confusion_matrix(official_values, predicted, labels=["RES", "NON_RES"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["RES", "NON_RES"], yticklabels=["RES", "NON_RES"])
    plt.title(f'Matriz de Confusion - {name}')
    plt.tight_layout()
    
    imagen_filename = os.path.join(output_path, f"{name}_cm.png")
    plt.savefig(imagen_filename, dpi=300)
    plt.close()

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "2. Matriz de Confusion", ln=True)
    pdf.image(imagen_filename, x=45, w=120) 
    pdf.ln(5)

    # 3. Análisis de Errores Críticos
    gdf_error = intersection[intersection['type'] != intersection['org_type']]
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "3. Top Errores de Clasificacion", ln=True)
    
    def agregar_errores_pdf(titulo, condicion):
        pdf.set_font("Helvetica", style="I", size=10)
        pdf.cell(0, 6, titulo, ln=True)
        pdf.set_font("Helvetica", size=10)
        errores = Counter(gdf_error[condicion]['tag used']).most_common(5)
        for tag, freq in errores:
            texto_tag = "Sin etiqueta en OSM (NaN)" if pd.isna(tag) else str(tag)
            pdf.cell(10)
            pdf.cell(0, 6, f"- {texto_tag}: {freq} veces", ln=True)
        pdf.ln(3)

    agregar_errores_pdf("a) Era NON_RES pero se clasifico como RES:", gdf_error['type']=='RES')
    agregar_errores_pdf("b) Era RES pero se clasifico como NON_RES:", gdf_error['type']=='NON_RES')

archivo_pdf = os.path.join(output_path, "Reporte_Completo.pdf")
pdf.output(archivo_pdf)
print(f"\n¡Exito! CSVs, GeoJSONs y el Reporte PDF guardados en: {output_path}")