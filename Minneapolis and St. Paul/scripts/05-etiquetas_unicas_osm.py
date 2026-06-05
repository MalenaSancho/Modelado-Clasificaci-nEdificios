import json
from google import genai
from google.genai import types
import os

API_KEY = 'PON TU API KEY AQUÍ'

# 1. Configura tu API Key
client = genai.Client(api_key=API_KEY)

# (Opcional) Listar modelos disponibles para asegurarnos de usar el correcto
# Descomenta las dos siguientes líneas si quieres ver qué modelos tienes habilitados:
# print("Modelos disponibles:")
# for m in client.models.list(): print(m.name)

# 2. Leer las etiquetas que extrajiste
try:
    with open("etiquetas_unicas_osm.txt", "r", encoding="utf-8") as f:
        etiquetas = [line.strip() for line in f.readlines() if line.strip()]
except FileNotFoundError:
    print("Error: No se encuentra el archivo 'etiquetas_unicas_osm.txt'.")
    exit()

print(f"Clasificando {len(etiquetas)} etiquetas con Gemini...")

# 3. El Prompt
prompt = f"""
Actúa como un experto en urbanismo, sistemas de información geográfica (SIG) y OpenStreetMap.
Estoy replicando la metodología del artículo científico "An OpenStreetMap derived building classification dataset for the United States" (de Arruda et al., 2024). 
Tengo una lista de etiquetas (tags) extraídas de OSM. Suelen tener el formato "categoría:valor" o simplemente "valor". 
CONTEXTO CRÍTICO: Todas estas etiquetas están asociadas a polígonos que ya sabemos que son EDIFICIOS (building footprints).

Necesito que clasifiques cada etiqueta estrictamente en una de estas tres categorías:

- "RES" (Residencial): Incluye viviendas (house, apartments, detached, etc.). CRÍTICO: Las estructuras secundarias asociadas a una casa, como garajes (garage, garages) y cobertizos (shed), DEBEN ser "RES".
- "NON_RES" (No Residencial): Incluye comercio, industria, oficinas, escuelas, hospitales, servicios públicos, etc. 
  REGLA DE PREFIJOS PARA NON_RES: Si la etiqueta contiene o empieza por categorías de infraestructura, servicios o comercio como "sport:", "leisure:", "tourism:", "transport:", "public_transport:", "amenity:", "shop:", "office:", o "craft:", clasifícala SIEMPRE como "NON_RES". No ignores la palabra antes de los dos puntos. (Ejemplo: "transport:platform" o "sport:baseball" son instalaciones cubiertas o asociadas a un edificio, por tanto son "NON_RES"). Los hoteles (hotel, motel) también son "NON_RES".
- "N/A" (Indeterminado / Ignorado): Etiquetas puramente estructurales o genéricas ("yes", "service", "roof", "ruins", "construction") o usos mixtos ("mixed_use") que no permitan decantarse por RES o NON_RES.

Analiza toda la cadena de texto de cada etiqueta y devuelve ÚNICAMENTE un objeto JSON válido donde la clave sea la etiqueta y el valor sea la categoría ("RES", "NON_RES" o "N/A"). No añadas texto explicativo ni bloques de código markdown, solo el JSON crudo.

Lista de etiquetas: {etiquetas}
"""

# 4. Llamada al LLM de Google forzando salida JSON
try:
    # Usamos gemini-3.5-flash 
    response = client.models.generate_content(
        model='gemini-3.5-flash', 
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )

    # 5. Guardar el diccionario generado
    diccionario_llm = json.loads(response.text)
    
    with open("diccionario_etiquetas_llm.json", "w", encoding="utf-8") as f:
        json.dump(diccionario_llm, f, indent=4, ensure_ascii=False)
        
    print("¡Éxito! Diccionario generado y guardado como 'diccionario_etiquetas_llm.json'.")

except Exception as e:
    print(f"Ocurrió un error: {e}")
    if 'response' in locals() and hasattr(response, 'text'):
        print("Respuesta cruda del modelo:")
        print(response.text)