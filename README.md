# Clasificación de edificios a partir de OpenStreetMap

Este repositorio contiene una adaptación y ampliación de una metodología de clasificación de edificios a partir de datos de **OpenStreetMap (OSM)**. El objetivo principal es clasificar huellas de edificios en dos clases:

- `RES`: edificios o zonas de carácter residencial.
- `NON_RES`: edificios o zonas de carácter no residencial.

El repositorio contiene dos casos de estudio principales:

1. **Minneapolis and St. Paul**, basado en el dataset oficial `GeneralizedLandUse2020` y dividido por condados.
2. **Alicante**, basado en SIOSE 2015/2014 de la Comunitat Valenciana y en los límites municipales oficiales del IGN/CNIG.

Además, el repositorio incluye una segunda fase experimental en la que se extraen las etiquetas únicas de OSM y se utiliza un LLM para construir un diccionario auxiliar de reclasificación de etiquetas.

---

## 1. Estructura general del repositorio

La estructura principal del proyecto es la siguiente:

```text
Modelado-Clasificaci-nEdificios/
│
├── Alicante/
│   ├── data/
│   ├── raw_data/
│   ├── out/
│   ├── out2/
│   ├── osmnx_cache/
│   ├── osmnx_cache2/
│   ├── scripts/
│   └── src/
│
├── Minneapolis and St. Paul/
    ├── data/
    ├── raw_data/
    ├── out/
    ├── out2/
    ├── osmnx_cache/
    ├── scripts/
    └── src/

```

La carpeta `Alicante/` contiene el flujo completo para generar un ground truth local de la ciudad de Alicante, validar la clasificación de edificios descargados desde OSM, generar informes y aplicar una versión mejorada con ayuda de un LLM.

La carpeta `Minneapolis and St. Paul/` contiene el flujo equivalente para el área metropolitana de Minneapolis and St. Paul, utilizando el dataset `GeneralizedLandUse2020` como fuente oficial de validación.

---

## 2. Dependencias principales

El proyecto trabaja con datos geoespaciales, descarga de datos desde OpenStreetMap, validación mediante métricas de clasificación y generación de informes. Las principales librerías utilizadas son:

```text
geopandas
pandas
shapely
osmnx
scikit-learn
tqdm
matplotlib
seaborn
fpdf
google-genai
```

Una instalación típica sería:

```bash
pip install -r requirements.txt
```


---

## 3. Datos necesarios

Los datos grandes no deberían subirse al repositorio. Deben descargarse y colocarse manualmente en las carpetas `raw_data/` correspondientes.

---

### 3.1. Datos para Minneapolis and St. Paul

Los datos necesarios para la parte de Minneapolis and St. Paul son los siguientes.

#### 3.1.1. Generalized Land Use 2020

Dataset principal de uso del suelo para la región metropolitana de Minneapolis and St. Paul.

Descargar en:

[Minneapolis and St. Paul](https://gisdata.mn.gov/dataset/us-mn-state-metc-plan-generl-lnduse2020) (```Metropolitan_reg_Minneapolis_and_St_Paul/GeneralizedLandUse2020.shp```)

Hay que escoger el archivo ```text SHAPEFILE ```.

Ruta esperada dentro del proyecto:

```text
Minneapolis and St. Paul\raw_data\Metropolitan_reg_Minneapolis_and_St_Paul
```

Este dataset contiene la columna `DESC2020`, que se reclasifica a:

```text
RES
NON_RES
N/A
```

Ejemplos de reclasificación:

```text
Single Family Detached     -> RES
Single Family Attached     -> RES
Multifamily                -> RES
Manufactured Housing Park  -> RES
Industrial or Utility      -> NON_RES
Institutional              -> NON_RES
Office                     -> NON_RES
Retail and Other Commercial -> NON_RES
Agricultural               -> N/A
Open Water                 -> N/A
Major Highway              -> N/A
Major Railway              -> N/A
```

---

#### 3.1.2. Counties 1:500,000 national

Shapefile de condados de Estados Unidos. Es necesario para dividir `GeneralizedLandUse2020` por condados y generar los conjuntos de validación independientes para cada condado del área metropolitana de Minneapolis and St. Paul.

Descargar en:

[Counties 1:500,000 (national)](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2023.html#list-tab-1883739534) (`cb_2023_us_county_500k/cb_2023_us_county_500k.shp`)

Dentro de la página del Census Bureau hay que ir a:

```text
Cartographic Boundary Files by Geography
Counties
1 : 500,000 (national) shapefile500,000 (national)   shapefile [313 MB]
```

Ruta esperada dentro del proyecto:

```text
Minneapolis and St. Paul\raw_data\cb_2023_us_county_500k
```

En el caso de Minneapolis and St. Paul, se extraen los siguientes condados:

```text
Anoka_MN
Carver_MN
Dakota_MN
Hennepin_MN
Ramsey_MN
Scott_MN
Washington_MN
```

---

### 3.2. Datos para Alicante

Para Alicante se utilizan dos fuentes de datos:

1. SIOSE 2015/2014 de la Comunitat Valenciana.
2. Límites municipales oficiales del IGN/CNIG.

---

#### 3.2.1. SIOSE 2015/2014 Comunitat Valenciana

Descarga directa:

[SIOSE](https://descargas.icv.gva.es/dcd/04_cubierta_usosuelo/SIOSE/2015_SIOSE0250/SIOSE2015_SHP.zip?tipo=directa&formato=shp)

Al descomprimir el ZIP deben aparecer archivos como:

```text
T_POLIGONOS.shp
T_POLIGONOS.dbf
T_POLIGONOS.shx
T_POLIGONOS.prj
T_VALORES.dbf
TC_SIOSE_COBERTURAS.dbf
TC_SIOSE_ATRIBUTOS.dbf
```

Ruta esperada dentro del proyecto:

```text
Alicante/raw_data/siose_cv
```

En este proyecto se utiliza principalmente:

- `T_POLIGONOS.shp`: geometrías SIOSE.
- `T_VALORES.dbf`: coberturas asociadas a cada polígono.
- `TC_SIOSE_COBERTURAS.dbf`: diccionario de coberturas SIOSE.

---

#### 3.2.2. Límites municipales del IGN/CNIG

Página de búsqueda del Centro de Descargas del CNIG:

[Centro de descargas](https://centrodedescargas.cnig.es/CentroDescargas/resultados-busqueda)


Producto que hay que descargar:

```text
Límites y Unidades Administrativas
Formato: SHAPEFILE
```

Dentro del ZIP descargado, la capa que interesa para Alicante es la de **recintos municipales** de península y Baleares en ETRS89:

```text
SHP_ETRS89/recintos_municipales_inspire_peninbal_etrs89/recintos_municipales_inspire_peninbal_etrs89.shp
```

Ruta esperada dentro del proyecto:

```text
Alicante/raw_data/limites_municipales/recintos_municipales_inspire_peninbal_etrs89
```

El municipio de Alicante aparece en la columna:

```text
NAMEUNIT
```

con valor:

```text
Alacant/Alicante
```

---

## 4. Funcionamiento general del método

El flujo general del repositorio es:

1. Preparar un **ground truth oficial** a partir de un dataset externo de uso del suelo.
2. Descargar edificios desde OpenStreetMap con `osmnx`.
3. Clasificar cada edificio OSM como `RES` o `NON_RES` mediante reglas basadas en etiquetas OSM.
4. Cruzar espacialmente los edificios OSM clasificados con el ground truth.
5. Calcular métricas de clasificación.
6. Generar informes en CSV, GeoJSON y PDF.
7. Mejorar la clasificación usando un diccionario de etiquetas generado por un LLM.

---

## 5. Carpeta `src/`

Cada caso de estudio contiene una carpeta `src/` con código auxiliar.

---

### 5.1. `src/map_buildings.py`

Este archivo contiene la lógica central de clasificación de edificios OSM.

Sus principales responsabilidades son:

- Definir qué etiquetas OSM se descargan.
- Separar footprints de edificios y objetos auxiliares.
- Clasificar edificios en `RES` o `NON_RES` mediante reglas.
- Usar información auxiliar que intersecta con los edificios.
- Descargar datos OSM por segmentos para evitar consultas demasiado grandes a Overpass.

Las etiquetas OSM se mantienen en inglés porque OpenStreetMap trabaja con claves y valores como:

```text
building:house
building:apartments
building:commercial
amenity:school
shop:supermarket
office:yes
landuse:residential
landuse:industrial
```

Aunque el caso de Alicante use ground truth en español, la clasificación OSM sigue dependiendo de etiquetas OSM en inglés.

---

### 5.2. `src/utils.py`

Contiene funciones auxiliares usadas por los scripts:

- `create_directory_if_not_exists(path)`: crea carpetas de salida si no existen.
- `read_shapefile_regions(...)`: lectura de regiones CBSA.
- `read_counties(...)`: lectura del shapefile nacional de condados de Estados Unidos.
- `get_counties_region(...)`: obtiene condados dentro de una región.
- `get_utm_crs_from_geodataframe(gdf)`: calcula un CRS UTM adecuado para trabajar con áreas e intersecciones.

---

## 6. Flujo de ejecución para Alicante

Todos los comandos de esta sección deben ejecutarse desde:

```text
Alicante/scripts/
```

---

### 6.1. Paso 0: inspeccionar SIOSE Alicante

Script:

```text
00-inspect_siose_alicante.py
```

Ejecutar:

```bash
python 00-inspect_siose_alicante.py
```

Este script:

1. Lee los límites municipales del IGN/CNIG.
2. Filtra el municipio `Alacant/Alicante`.
3. Lee `T_POLIGONOS.shp` de SIOSE.
4. Recorta SIOSE al municipio de Alicante.
5. Lee `T_VALORES.dbf` y `TC_SIOSE_COBERTURAS.dbf`.
6. Obtiene la cobertura dominante por polígono.
7. Genera archivos de inspección.

Salidas principales:

```text
Alicante/out/alicante_siose_inspection/siose_alicante_polygons.gpkg
Alicante/out/alicante_siose_inspection/siose_alicante_all_components.csv
Alicante/out/alicante_siose_inspection/siose_alicante_dominant_coverages.csv
Alicante/out/alicante_siose_inspection/siose_alicante_coverage_summary.csv
```

Este paso sirve para comprobar qué coberturas aparecen en Alicante antes de crear el ground truth binario.

---

### 6.2. Paso 1: crear ground truth de Alicante

Script:

```text
01-cretate_ground_truth.py
```

Ejecutar:

```bash
python 01-cretate_ground_truth.py
```

Este script:

1. Lee los límites municipales.
2. Filtra `Alacant/Alicante`.
3. Lee SIOSE.
4. Recorta SIOSE al municipio.
5. Obtiene la cobertura dominante de cada polígono.
6. Reclasifica las categorías SIOSE en `RES`, `NON_RES` o `N/A`.
7. Elimina las categorías `N/A`.
8. Guarda el ground truth final.

La salida principal es:

```text
Alicante/data/test_region2gdf_alicante.pickle
```

También genera:

```text
Alicante/out/alicante_ground_truth/alicante_ground_truth.gpkg
Alicante/out/alicante_ground_truth/alicante_ground_truth_summary.csv
Alicante/out/alicante_ground_truth/alicante_ground_truth_all_polygons.csv
```

El pickle generado tiene la estructura:

```python
name2gdf = {
    "Alicante": gdf_alicante
}
```

con columnas principales:

```text
geometry
org_type
old_label
ID_POLYGON
SIOSE_CODE
ID_COBER
CODE_ABREV
DESC_COBER
clip_area_ha
```

---

### 6.3. Paso 2: validar Alicante con el modelo base

Script:

```text
02-validate_alicante.py
```

Ejecutar:

```bash
python 02-validate_alicante.py
```

Este script:

1. Carga `data/test_region2gdf_alicante.pickle`.
2. Descarga edificios de OpenStreetMap para Alicante mediante `osmnx`.
3. Clasifica edificios con `src/map_buildings.py`.
4. Calcula el mayor solapamiento entre cada edificio OSM y el ground truth SIOSE.
5. Calcula métricas de clasificación.
6. Guarda los resultados en pickle.

Salidas:

```text
Alicante/out/comparison_alicante/name2performance_alicante.pickle
Alicante/out/comparison_alicante/name2identified_gdf_alicante.pickle
Alicante/out/comparison_alicante/name2intersection_alicante.pickle
```

`name2identified_gdf_alicante.pickle` contiene los edificios descargados y clasificados desde OSM.

`name2intersection_alicante.pickle` contiene únicamente los edificios que se han podido cruzar con el ground truth.

---

### 6.4. Paso 3: generar informe base de Alicante

Script:

```text
03-reporte_final.py
```

Ejecutar:

```bash
python 03-reporte_final.py
```

Este script toma los resultados del paso anterior y genera:

```text
Alicante/out/reporte_final/Alicante_resultados.csv
Alicante/out/reporte_final/Alicante_mapa.geojson
Alicante/out/reporte_final/Alicante_cm.png
Alicante/out/reporte_final/Reporte_Completo.pdf
```

El PDF contiene:

- Métricas de clasificación.
- Matriz de confusión.
- Principales errores según etiquetas OSM.

---

### 6.5. Paso 4: extraer etiquetas únicas de OSM

Script:

```text
04-extraer_etiquetas.py
```

Ejecutar:

```bash
python 04-extraer_etiquetas.py
```

Este script lee:

```text
Alicante/out/comparison_alicante/name2identified_gdf_alicante.pickle
```

y extrae todos los valores únicos de la columna:

```text
tag used
```

Salida:

```text
Alicante/scripts/etiquetas_unicas_osm_alicante.txt
```

Este archivo se usa como entrada para el LLM.

---

### 6.6. Paso 5: clasificar etiquetas únicas con LLM

Script:

```text
05-etiquetas_unicas_osm.py
```

Ejecutar:

```bash
python 05-etiquetas_unicas_osm.py
```

Antes de ejecutarlo, hay que configurar la API key (ir al final del README.md):

```python
API_KEY = 'PON TU API KEY AQUÍ'
```

Este script:

1. Lee `etiquetas_unicas_osm_alicante.txt`.
2. Envía las etiquetas por lotes a Gemini.
3. Pide al modelo que clasifique cada etiqueta como `RES`, `NON_RES` o `N/A`.
4. Guarda un diccionario JSON.

Salida:

```text
Alicante/scripts/diccionario_etiquetas_llm_alicante.json
```

Este diccionario no sustituye el ground truth. Solo se usa para mejorar la clasificación de las etiquetas OSM.

---

### 6.7. Paso 6: validar Alicante con mejora LLM + área

Script:

```text
06-validate2.py
```

Ejecutar:

```bash
python 06-validate2.py
```

Este script repite la validación de Alicante, pero añade una corrección posterior:

1. Si un edificio no tiene etiqueta clara, usa el área:
   - área mayor que `800 m²` → `NON_RES`.
   - área menor o igual que `800 m²` → `RES`.
2. Si el edificio tiene una etiqueta incluida en `diccionario_etiquetas_llm_alicante.json`, usa la clasificación del LLM.
3. Si el LLM devuelve `N/A` o no conoce la etiqueta, mantiene la clasificación original del modelo base.

Salidas:

```text
Alicante/out2/comparison2_alicante/name2performance_alicante.pickle
Alicante/out2/comparison2_alicante/name2identified_gdf_alicante.pickle
Alicante/out2/comparison2_alicante/name2intersection_alicante.pickle
```

---

### 6.8. Paso 7: generar informe mejorado de Alicante

Script:

```text
07-reporte_final2.py
```

Ejecutar:

```bash
python 07-reporte_final2.py
```

Este script genera el informe final de la validación mejorada.

Salidas:

```text
Alicante/out2/reporte_final2/Alicante_resultados.csv
Alicante/out2/reporte_final2/Alicante_mapa.geojson
Alicante/out2/reporte_final2/Alicante_cm.png
Alicante/out2/reporte_final2/Reporte_Completo.pdf
```

---

## 7. Flujo de ejecución para Minneapolis and St. Paul

Todos los comandos de esta sección deben ejecutarse desde:

```text
Minneapolis and St. Paul/scripts/
```

---

### 7.1. Paso 1: crear ground truth de Minneapolis and St. Paul

Script:

```text
01-cretate_ground_truth.py
```

Ejecutar:

```bash
python 01-cretate_ground_truth.py
```

Este script:

1. Lee `GeneralizedLandUse2020.shp`.
2. Reclasifica `DESC2020` a `RES`, `NON_RES` o `N/A`.
3. Elimina categorías `N/A`.
4. Elimina solapamientos entre clases.
5. Lee el shapefile nacional de condados.
6. Hace un `spatial join` para separar la región en siete condados.
7. Guarda un pickle con los GeoDataFrames oficiales por condado.

Salida principal:

```text
Minneapolis and St. Paul/data/test_region2gdf.pickle
```

El pickle tiene la estructura:

```python
name2gdf = {
    "Anoka_MN": gdf,
    "Carver_MN": gdf,
    "Dakota_MN": gdf,
    "Hennepin_MN": gdf,
    "Ramsey_MN": gdf,
    "Scott_MN": gdf,
    "Washington_MN": gdf,
}
```

---

### 7.2. Paso 2: validar con el modelo base

Script:

```text
02-validate.py
```

Ejecutar:

```bash
python 02-validate.py
```

Este script:

1. Carga `data/test_region2gdf.pickle`.
2. Descarga edificios OSM por condado.
3. Clasifica los edificios con `src/map_buildings.py`.
4. Cruza edificios OSM con el ground truth oficial.
5. Calcula métricas por condado.
6. Guarda resultados.

Salidas:

```text
Minneapolis and St. Paul/out/comparison/name2performance.pickle
Minneapolis and St. Paul/out/comparison/name2identified_gdf.pickle
Minneapolis and St. Paul/out/comparison/name2intersection.pickle
```

---

### 7.3. Paso 3: generar informe base

Script:

```text
03-reporte_final.py
```

Ejecutar:

```bash
python 03-reporte_final.py
```

Salidas:

```text
Minneapolis and St. Paul/out/reporte_final/<CONDADO>_resultados.csv
Minneapolis and St. Paul/out/reporte_final/<CONDADO>_mapa.geojson
Minneapolis and St. Paul/out/reporte_final/<CONDADO>_cm.png
Minneapolis and St. Paul/out/reporte_final/Reporte_Completo.pdf
```

---

### 7.4. Paso 4: extraer etiquetas únicas

Script:

```text
04-extraer_etiquetas.py
```

Ejecutar:

```bash
python 04-extraer_etiquetas.py
```

Salida:

```text
Minneapolis and St. Paul/scripts/etiquetas_unicas_osm.txt
```

---

### 7.5. Paso 5: clasificar etiquetas con LLM

Script:

```text
05-etiquetas_unicas_osm.py
```

Ejecutar:

```bash
python 05-etiquetas_unicas_osm.py
```

Antes de ejecutarlo, hay que configurar la API key:

```python
API_KEY = 'PON TU API KEY AQUÍ'
```

Salida:

```text
Minneapolis and St. Paul/scripts/diccionario_etiquetas_llm.json
```

---

### 7.6. Paso 6: aplicar mejora LLM + área

Script:

```text
06-validate2.py
```

Ejecutar:

```bash
python 06-validate2.py
```

Este script no vuelve a descargar datos OSM. Carga las intersecciones ya calculadas en el paso 2 y aplica la mejora basada en:

- diccionario LLM de etiquetas;
- umbral de área para edificios sin etiqueta.

Salidas:

```text
Minneapolis and St. Paul/out2/comparison2/name2performance.pickle
Minneapolis and St. Paul/out2/comparison2/name2intersection.pickle
```

---

### 7.7. Paso 7: generar informe mejorado

Script:

```text
07-reporte_final2.py
```

Ejecutar:

```bash
python 07-reporte_final2.py
```

Salidas:

```text
Minneapolis and St. Paul/out2/reporte_final2/<CONDADO>_resultados.csv
Minneapolis and St. Paul/out2/reporte_final2/<CONDADO>_mapa.geojson
Minneapolis and St. Paul/out2/reporte_final2/<CONDADO>_cm.png
Minneapolis and St. Paul/out2/reporte_final2/Reporte_Completo.pdf
```

---

## 8. Orden de ejecución recomendado

### Alicante

```bash
cd "Alicante/scripts"

python 00-inspect_siose_alicante.py
python 01-cretate_ground_truth.py
python 02-validate_alicante.py
python 03-reporte_final.py
python 04-extraer_etiquetas.py
python 05-etiquetas_unicas_osm.py
python 06-validate2.py
python 07-reporte_final2.py
```

### Minneapolis and St. Paul

```bash
cd "Minneapolis and St. Paul/scripts"

python 01-cretate_ground_truth.py
python 02-validate.py
python 03-reporte_final.py
python 04-extraer_etiquetas.py
python 05-etiquetas_unicas_osm.py
python 06-validate2.py
python 07-reporte_final2.py
```

---

## 9. Carpetas de salida

### `data/`

Contiene los ground truth procesados en formato pickle.

Ejemplos:

```text
Alicante/data/test_region2gdf_alicante.pickle
Minneapolis and St. Paul/data/test_region2gdf.pickle
```

---

### `out/`

Contiene resultados de la validación base.

Ejemplos:

```text
out/comparison*/
out/reporte_final/
out/alicante_ground_truth/
out/alicante_siose_inspection/
```

---

### `out2/`

Contiene resultados de la validación mejorada con LLM y análisis geométrico por área.

Ejemplos:

```text
out2/comparison2*/
out2/reporte_final2/
```

---

### `osmnx_cache/` y `osmnx_cache2/`

Carpetas de caché utilizadas por OSMnx para no repetir descargas ya realizadas desde Overpass.

Estas carpetas pueden ocupar bastante espacio y no deberían subirse al repositorio.

---

## 10. Sobre OpenStreetMap y Overpass

Los scripts de validación descargan datos desde OpenStreetMap usando `osmnx.features_from_polygon(...)`.

Para evitar consultas demasiado grandes, el área de estudio se divide en segmentos mediante:

```python
generate_gdf_with_segments(polygon, num_segments, map_buildings.tags)
```

Si Overpass falla por timeout, se puede aumentar el número de segmentos.

Ejemplo:

```python
num_segments = 3
```

puede cambiarse por:

```python
num_segments = 5
```

También se puede cambiar el servidor Overpass descomentando:

```python
ox.settings.overpass_url = "https://overpass.kumi.systems/api"
```

---

## 11. Diferencia entre ground truth y datos OSM

El ground truth es la verdad oficial contra la que se valida el modelo.

En Minneapolis and St. Paul, el ground truth se crea desde:

```text
GeneralizedLandUse2020.shp
```

En Alicante, el ground truth se crea desde:

```text
SIOSE 2015/2014 + límites municipales oficiales
```

Los datos OSM son los edificios que el modelo clasifica. Estos se descargan en los scripts de validación, no forman parte del ground truth.

Por tanto, el flujo conceptual es:

```text
Ground truth oficial
        ↓
Polígonos RES / NON_RES
        ↓
Descarga de edificios OSM
        ↓
Clasificación OSM RES / NON_RES
        ↓
Cruce espacial
        ↓
Métricas
```

---

## 12. Clasificación base de edificios OSM

La clasificación base se realiza en `src/map_buildings.py`.

Ejemplos de reglas:

```text
building:house        -> RES
building:apartments   -> RES
building:detached     -> RES
building:garage       -> RES
building:shed         -> RES
building:commercial   -> NON_RES
building:industrial   -> NON_RES
amenity:school        -> NON_RES
amenity:hospital      -> NON_RES
shop:*                -> NON_RES
office:*              -> NON_RES
```

Cuando un edificio tiene una etiqueta genérica o desconocida, el modelo base tiende a clasificarlo como residencial:

```text
residential_unknown_tag -> RES
```

La versión mejorada intenta corregir algunos de estos casos con un diccionario LLM y un umbral de área.

---

## 13. Clasificación mejorada con LLM

La clasificación mejorada sigue estos pasos:

1. Ejecutar la validación base.
2. Extraer las etiquetas únicas de OSM.
3. Pedir a un LLM que clasifique esas etiquetas en `RES`, `NON_RES` o `N/A`.
4. Guardar el resultado en un JSON.
5. Reaplicar la clasificación usando ese JSON.

En Alicante, el JSON esperado es:

```text
Alicante/scripts/diccionario_etiquetas_llm_alicante.json
```

En Minneapolis and St. Paul, el JSON esperado es:

```text
Minneapolis and St. Paul/scripts/diccionario_etiquetas_llm.json
```

La mejora se aplica así:

- Si el edificio no tiene etiqueta clara, se usa el área.
- Si el edificio tiene etiqueta y el LLM la clasifica como `RES` o `NON_RES`, se usa esa clase.
- Si el LLM devuelve `N/A`, se conserva la clasificación base.

## 14. Cómo conseguir la API KEY

1. Dirígete a este enlace: [API KEY](https://aistudio.google.com/app/api-keys?)
2. Selecciona *Crear clave de API*.
3. En *Elige un proyecto importado*, escribe *Default Gemini Project*.
4. Pulsa *Crear clave*.
5. Copia *Clave de API* y sustitúyela en los scripts *05-etiquetas_unicas_osm.py*.


## 15. Citas

El trabajo toma como referencia principal el artículo original en el que se presenta el conjunto de datos de clasificación de edificios derivado de OpenStreetMap para Estados Unidos. A partir de esta publicación hemos estructurado el enfoque general del proyecto, tanto en la descarga y procesamiento de datos geoespaciales como en la clasificación de edificios en categorías residenciales y no residenciales.

La referencia bibliográfica principal es:

F. de Arruda, H., Reia, S.M., Ruan, S. et al. An OpenStreetMap derived building classification dataset for the United States. Sci Data 11, 1210 (2024). https://doi.org/10.1038/s41597-024-04046-w