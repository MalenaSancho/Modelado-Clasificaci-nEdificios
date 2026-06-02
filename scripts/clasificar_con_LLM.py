import requests
import time
import json
import os

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
# Pega tu token aquí dentro:

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
CATEGORIAS = ["RES", "NON_RES", "UNKNOWN"]

archivo_entrada = "etiquetas_unicas_osm.txt"
archivo_salida = "diccionario_etiquetas_llm.json"

# ==========================================
# 2. FUNCIÓN DE CLASIFICACIÓN (Súper blindada)
# ==========================================
def clasificar_etiqueta(etiqueta, max_reintentos=10):
    payload = {
        "inputs": etiqueta,
        "parameters": {"candidate_labels": CATEGORIAS}
    }
    
    for intento in range(max_reintentos):
        try:
            # Ponemos un timeout de 15 segundos para que no se quede colgado eternamente
            response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                resultado = response.json()
                return resultado['labels'][0]
            
            elif response.status_code == 503:
                print(f"\n  [!] El modelo está cargando en Hugging Face. Esperando 10 segundos... (Intento {intento+1}/{max_reintentos})")
                time.sleep(10)
            else:
                print(f"\n  [X] Error inesperado con la etiqueta '{etiqueta}': {response.text}")
                break
                
        # Esta captura TODOS los errores de red (desconexión, DNS, timeout...)
        except requests.exceptions.RequestException as e:
            print(f"\n  [!] Micro-corte de red detectado. Reintentando en 5 segundos... (Intento {intento+1}/{max_reintentos})")
            time.sleep(5)
            
    return "UNKNOWN"

# ==========================================
# 3. EJECUCIÓN PRINCIPAL
# ==========================================
print(f"Leyendo etiquetas desde: {archivo_entrada}...")
with open(archivo_entrada, "r", encoding="utf-8") as f:
    etiquetas = [line.strip() for line in f if line.strip()]

print(f"Se encontraron {len(etiquetas)} etiquetas para clasificar.\n")

diccionario_resultados = {}

for i, etiqueta in enumerate(etiquetas):
    print(f"[{i+1}/{len(etiquetas)}] Clasificando: {etiqueta}...", end=" ", flush=True)
    
    categoria = clasificar_etiqueta(etiqueta)
    diccionario_resultados[etiqueta] = categoria
    
    print(f"-> {categoria}")
    time.sleep(1) # Pausa obligatoria para no saturar la API gratuita

# ==========================================
# 4. GUARDAR RESULTADOS
# ==========================================
with open(archivo_salida, "w", encoding="utf-8") as f:
    json.dump(diccionario_resultados, f, indent=4, ensure_ascii=False)

print(f"\n¡Proceso completado! Los resultados se han guardado en: {archivo_salida}")