import json
import os
import sys
import time
from google import genai
from google.genai import types

API_KEY = 'PON TU API KEY AQUÍ'

# 1. Configura tu API Key
client = genai.Client(api_key=API_KEY)

# 2. Leer las etiquetas
try:
    with open("etiquetas_unicas_osm_alicante.txt", "r", encoding="utf-8") as f:
        etiquetas = [line.strip() for line in f.readlines() if line.strip()]
except FileNotFoundError:
    print("Error: No se encuentra el archivo 'etiquetas_unicas_osm_alicante.txt'.")
    sys.exit()

print(f"Total de etiquetas cargadas: {len(etiquetas)}")

# 3. Función para clasificar un lote de etiquetas
def clasificar_lote(lote_etiquetas):
    prompt = f"""
    Actúa como un experto en urbanismo, sistemas de información geográfica (SIG) y OpenStreetMap.
    Estoy replicando la metodología del artículo científico "An OpenStreetMap derived building classification dataset for the United States" (de Arruda et al., 2024). 
    Tengo una lista de etiquetas (tags) extraídas de OSM. Suelen tener el formato "categoría:valor" o simplemente "valor". 
    CONTEXTO CRÍTICO: Todas estas etiquetas están asociadas a polígonos que ya sabemos que son EDIFICIOS (building footprints).

    Necesito que clasifiques cada etiqueta estrictamente en una de estas tres categorías:

    - "RES" (Residencial): Incluye viviendas (house, apartments, detached, yards, etc.). CRÍTICO: Las estructuras secundarias asociadas a una casa, como garajes (garage, garages) y cobertizos (shed), DEBEN ser "RES".
    - "NON_RES" (No Residencial): Incluye comercio, industria, oficinas, escuelas, hospitales, servicios públicos, parkings, accesos a sitios públicos, etc. 
      REGLA DE PREFIJOS PARA NON_RES: Si la etiqueta contiene o empieza por categorías de infraestructura, servicios o comercio como "sport:", "leisure:", "tourism:", "transport:", "public_transport:", "amenity:", "shop:", "office:", o "craft:", clasifícala SIEMPRE como "NON_RES". No ignores la palabra antes de los dos puntos. (Ejemplo: "transport:platform" o "sport:baseball" son instalaciones cubiertas o asociadas a un edificio, por tanto son "NON_RES"). Los hoteles (hotel, motel) también son "NON_RES".
    - "N/A" (Indeterminado / Ignorado): Etiquetas puramente estructurales o genéricas ("yes", "service", "roof", "ruins", "construction") o usos mixtos ("mixed_use") que no permitan decantarse por RES o NON_RES.

    Analiza toda la cadena de texto de cada etiqueta y devuelve ÚNICAMENTE un objeto JSON válido donde la clave sea la etiqueta y el valor sea la categoría ("RES", "NON_RES" o "N/A"). No añadas texto explicativo ni bloques de código markdown, solo el JSON crudo.

    Lista de etiquetas a clasificar en este lote: {lote_etiquetas}
    """

    # Forzamos un timeout alto de 120 segundos para que no caiga la conexión
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1 # Temperatura baja para que sea determinista y estricto
    )
    
    # IMPORTANTE: Cambiado a un modelo real -> 'gemini-2.5-flash'
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt,
        config=config
    )
    
    return json.loads(response.text)

# 4. Procesamiento por lotes (Evita congelación y sobrecarga del JSON de salida)
diccionario_llm_final = {}
TAMAÑO_LOTE = 100

print("Iniciando clasificación por lotes con Gemini...")
for i in range(0, len(etiquetas), TAMAÑO_LOTE):
    lote = etiquetas[i:i + TAMAÑO_LOTE]
    print(f"-> Procesando lote {i // TAMAÑO_LOTE + 1} (Etiquetas {i} a {i + len(lote)})...")
    
    intentos = 3
    while intentos > 0:
        try:
            resultado_lote = clasificar_lote(lote)
            diccionario_llm_final.update(resultado_lote)
            break
        except Exception as e:
            intentos -= 1
            print(f"   Error en el lote (quedan {intentos} intentos): {e}")
            if intentos > 0:
                time.sleep(5) # Espera de cortesía antes de reintentar
            else:
                print("❌ No se pudo completar este bloque tras varios intentos.")

# 5. Guardar el archivo definitivo unificado
with open("diccionario_etiquetas_llm_alicante.json", "w", encoding="utf-8") as f:
    json.dump(diccionario_llm_final, f, indent=4, ensure_ascii=False)
    
print(f"\n¡Éxito! Proceso finalizado. Se han clasificado {len(diccionario_llm_final)} etiquetas.")
print("Archivo guardado en: 'diccionario_etiquetas_llm_alicante.json'")