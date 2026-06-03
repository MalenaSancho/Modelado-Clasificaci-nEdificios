import pickle
import os
import pandas as pd

input_path = "../out/comparison"

print("Cargando los datos de los edificios...\n")
with open(os.path.join(input_path, 'name2identified_gdf.pickle'), 'rb') as handle:
    name2identified_gdf = pickle.load(handle)

todas_las_etiquetas = set()

# Recorremos todos los condados para extraer las etiquetas que encontró OSM
for condado, gdf in name2identified_gdf.items():
    if 'tag used' in gdf.columns:
        # Extraemos las etiquetas, quitamos los nulos (None) y sacamos los valores únicos
        etiquetas = gdf['tag used'].dropna().unique()
        todas_las_etiquetas.update(etiquetas)

# Ordenamos alfabéticamente para que quede limpio
lista_etiquetas = sorted(list(todas_las_etiquetas))

print(f"¡Éxito! Se han encontrado {len(lista_etiquetas)} etiquetas únicas en toda la zona metropolitana.")

# Guardamos la lista en un archivo de texto
output_file = "etiquetas_unicas_osm.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for tag in lista_etiquetas:
        f.write(f"{tag}\n")

print(f"\nLista guardada en el archivo: {output_file}")
print("Abre este archivo. Es lo que le pasaremos al LLM.")