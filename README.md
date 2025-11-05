# Sistema de Clasificación Inteligente
## Trabajo Final - Inteligencia Artificial I

### Descripción
Sistema de clasificación que integra:
- **Visión artificial** con K-Means para clasificar piezas (tornillos, clavos, arandelas, tuercas)
- **Reconocimiento de voz** con K-NN para comandos ("proporcion", "contar", "salir")
- **Aprendizaje bayesiano** para estimación de distribuciones

### Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐
│   Entrenamiento │    │   Monitoreo     │
│                 │    │   Continuo      │
├─────────────────┤    ├─────────────────┤
│ 1. Procesar     │    │ 1. img_new/     │
│    img2/ → out/ │    │    (imágenes)   │
│                 │    │                 │
│ 2. Extraer      │    │ 2. comando/     │
│    features     │    │    (audios)     │
│                 │    │                 │
│ 3. Entrenar     │    │ 3. Clasificar   │
│    KMeans/KNN   │    │    en tiempo    │
│                 │    │    real         │
│ 4. Guardar      │    │                 │
│    modelos      │    │ 4. Análisis     │
│                 │    │    bayesiano    │
└─────────────────┘    └─────────────────┘
```

### Estructura de Archivos

```
ProyectoFinalIA/
├── main.py                 # Punto de entrada principal
├── demo.py                 # Demostración del sistema
├── test_system.py          # Pruebas del sistema
├── requirements.txt        # Dependencias
│
├── modules/                # Módulos del sistema
│   ├── app_controller.py   # Controlador principal (NUEVO)
│   ├── binary.py           # Procesamiento de imágenes
│   ├── image_params.py     # Extracción de características
│   ├── audio_params.py     # Características de audio
│   ├── my_kmeans.py        # Algoritmo K-Means
│   ├── my_knn.py           # Algoritmo K-NN
│   └── bayes.py           # Clasificador bayesiano
│
├── img2/                   # Imágenes de entrenamiento
├── audio/                  # Audios de entrenamiento
├── comando/                # Audios nuevos (monitoreado)
├── img_new/                # Imágenes nuevas (monitoreado)
│
├── models/                 # Modelos entrenados (auto-generado)
│   ├── kmeans_model.pkl
│   ├── knn_model.pkl
│   └── audio_stats.json
│
└── temp_*/                 # Directorios temporales (auto-generado)
```

### Flujo de Ejecución

#### 1. Inicialización
```bash
python main.py
```

1. **Carga/Entrenamiento de Modelos**:
   - Verifica si existen modelos guardados
   - Si no existen, entrena desde datos en `img2/` y `audio/`
   - Guarda modelos en `models/`

2. **Inicio de Monitoreo**:
   - Configura observadores de archivos
   - Monitorea `comando/` y `img_new/`

#### 2. Clasificación en Tiempo Real

**Nuevas Imágenes** (`img_new/`):
1. Detecta archivo `.jpg/.png` nuevo
2. Procesa imagen (binarización, extracción de contornos)
3. Extrae características geométricas
4. Clasifica con modelo K-Means entrenado
5. Almacena resultado para análisis bayesiano

**Comandos de Audio** (`comando/`):
1. Detecta archivo `.wav` nuevo
2. Extrae características de audio (MFCC, ZCR, etc.)
3. Clasifica comando con modelo K-NN entrenado
4. Ejecuta acción correspondiente:
   - `"proporcion"` → Muestra distribución estimada
   - `"contar"` → Muestra conteo de 1000 piezas
   - `"salir"` → Finaliza sistema

#### 3. Análisis Bayesiano
- Usa las últimas 10 clasificaciones de imágenes como muestra
- Aplica modelo bayesiano con 4 cajas predefinidas:
  - Caja A: 250/250/250/250 (equilibrada)
  - Caja B: 150/300/300/250 (más clavos/arandelas)
  - Caja C: 250/350/250/150 (más clavos)
  - Caja D: 500/500/0/0 (solo tornillos/clavos)

### Uso del Sistema

#### Ejecución Normal
```bash
python main.py
```

#### Demostración
```bash
python demo.py
```

#### Pruebas del Sistema
```bash
python test_system.py
```

### Comandos de Voz Disponibles

| Comando | Acción |
|---------|--------|
| `proporcion` | Muestra distribución porcentual estimada |
| `contar` | Muestra conteo estimado de 1000 piezas |
| `salir` | Finaliza la aplicación |

### Ejemplo de Uso

1. **Iniciar sistema**:
   ```bash
   python main.py
   ```

2. **Agregar imágenes** (en otra terminal):
   ```bash
   cp imagen_tornillo.jpg img_new/
   ```
   
3. **Dar comando de voz**:
   ```bash
   cp audio_proporcion.wav comando/
   ```

4. **Ver resultados** en la terminal principal:
   ```
   🖼️ Nueva imagen detectada: imagen_tornillo.jpg
   🔍 Clasificación: tornillo
   🎵 Comando de audio detectado: audio_proporcion.wav
   🎯 Comando reconocido: proporcion
   
   📈 PROPORCIONES ESTIMADAS:
   ========================================
     tornillo    : 65.23%
     clavo       : 15.45%
     arandela    : 12.87%
     tuerca      :  6.45%
   ```

### Características Técnicas

#### Procesamiento de Imágenes
- **Preprocesamiento**: Filtrado gaussiano, umbralización adaptativa
- **Segmentación**: Componentes conectados, llenado de huecos
- **Características**: Relación área/círculo, momentos de Hu, ángulos, curvatura

#### Procesamiento de Audio
- **Frecuencia de muestreo**: 16kHz, mono
- **Características**: MFCC, Zero Crossing Rate, Spectral Rolloff
- **Normalización**: Z-score estandarizado

#### Algoritmos
- **K-Means**: Inicialización k-means++, manejo de clusters vacíos
- **K-NN**: Distancia Manhattan, pesos por distancia
- **Bayesiano**: Promedio de modelos con prior uniforme

### Extensibilidad

El sistema está diseñado para fácil extensión:

1. **Nuevas características**: Agregar en `image_params.py` o `audio_params.py`
2. **Nuevos comandos**: Modificar `execute_command()` en `app_controller.py`
3. **Nuevos modelos bayesianos**: Actualizar `_MODELS` en `bayes.py`

### Dependencias

```
opencv-python>=4.5.0
numpy>=1.20.0
scikit-learn>=1.0.0
librosa>=0.8.0
watchdog>=2.0.0
```

### Notas de Implementación

- **Modelos persistentes**: Se guardan automáticamente tras entrenamiento
- **Monitoreo robusto**: Usa watchdog para detectar archivos nuevos
- **Limpieza automática**: Elimina archivos temporales tras procesamiento
- **Manejo de errores**: Continúa funcionando ante archivos corruptos
- **Thread-safe**: Manejo seguro de concurrencia en monitoreo