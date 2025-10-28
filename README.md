# 🔧 Sistema de Reconocimiento de Elementos de Ferretería

python3 -m venv IA_env
source IA_env/bin/activate

pip install opencv-python numpy matplotlib

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de visión artificial para el reconocimiento automático de elementos de ferretería comunes: **tornillos**, **tuercas**, **arandelas** y **clavos**. El sistema está diseñado siguiendo los principios de la materia de Inteligencia Artificial, utilizando técnicas clásicas de procesamiento de imágenes sin recurrir a redes neuronales complejas.

## 🎯 Objetivos

- **Objetivo Principal**: Desarrollar un sistema capaz de identificar y clasificar elementos de ferretería en imágenes
- **Objetivos Específicos**:
  - Implementar un módulo robusto de adquisición y preprocesamiento de datos
  - Aplicar técnicas de filtrado y mejora de imágenes
  - Desarrollar algoritmos de segmentación y extracción de características
  - Crear un clasificador basado en características geométricas

## 🏗️ Estructura del Proyecto

```
ProyectoFinalIA/
├── src/                          # Código fuente principal
│   ├── adquisicion_datos.py       # Módulo de carga y preprocesamiento
│   ├── extraccion_caracteristicas.py  # (Próximo módulo)
│   └── clasificador.py            # (Próximo módulo)
├── data/                         # Datos del proyecto
│   ├── imagenes_originales/      # Imágenes de entrada
│   ├── imagenes_procesadas/      # Imágenes después del preprocesamiento
│   ├── modelos/                  # Modelos entrenados
│   └── resultados/               # Resultados de clasificación
├── tests/                        # Pruebas unitarias
├── pdf/                          # Documentación del proyecto
├── config.py                     # Configuración del sistema
├── ejemplos_uso.py               # Ejemplos de uso del sistema
└── requirements.txt              # Dependencias del proyecto
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.12+**: Lenguaje de programación principal
- **OpenCV**: Procesamiento de imágenes y visión artificial
- **NumPy**: Operaciones numéricas y matrices
- **Matplotlib**: Visualización de resultados

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd ProyectoFinalIA
```

### 2. Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar instalación
```bash
python config.py
```

## 🚀 Uso del Sistema

### Módulo de Adquisición y Preprocesamiento

El módulo `adquisicion_datos.py` es el primer componente del sistema. Proporciona funcionalidades para:

#### 1. Carga de Imágenes
```python
from src.adquisicion_datos import ProcesadorImagenes

# Crear procesador
procesador = ProcesadorImagenes()

# Cargar imagen
imagen = procesador.cargar_imagen("tornillo.jpg")
```

#### 2. Preprocesamiento Completo
```python
# Procesamiento completo con configuración automática
resultado = procesador.procesar_imagen_completa(
    "tornillo.jpg",
    tamaño=(512, 512),
    aplicar_filtro='gaussiano',
    metodo_binarizacion='otsu',
    guardar_pasos=True
)
```

#### 3. Procesamiento Paso a Paso
```python
# Preprocesamiento manual paso a paso
imagen_original = procesador.cargar_imagen("tuerca.jpg")
imagen_redim = procesador.redimensionar_imagen(imagen_original, (400, 400))
imagen_gris = procesador.convertir_escala_grises(imagen_redim)
imagen_filtrada = procesador.aplicar_filtro_gaussiano(imagen_gris)
imagen_bin = procesador.binarizar_imagen(imagen_filtrada, 'otsu')
bordes = procesador.detectar_bordes_canny(imagen_filtrada)
```

### Ejemplo Rápido

```bash
# Ejecutar ejemplo completo
python ejemplos_uso.py
```

## 🔍 Técnicas de Preprocesamiento Implementadas

### 1. **Redimensionamiento**
- Estandariza el tamaño de todas las imágenes
- Reduce el costo computacional
- Facilita el procesamiento posterior

### 2. **Conversión a Escala de Grises**
- Reduce la dimensionalidad (3 canales → 1 canal)
- Enfoca el análisis en forma y textura
- Simplifica algoritmos posteriores

### 3. **Filtrado para Eliminación de Ruido**

#### Filtro Gaussiano
- Suaviza la imagen preservando bordes importantes
- Reduce ruido de alta frecuencia
- Configurable mediante tamaño de kernel y sigma

#### Filtro de Mediana
- Elimina ruido tipo "sal y pimienta"
- Preserva bordes mejor que filtros lineales
- Ideal para ruido impulsivo

### 4. **Binarización**

#### Método de Otsu
- Calcula automáticamente el umbral óptimo
- Minimiza la varianza intra-clase
- Ideal para imágenes con bimodalidad clara

#### Binarización Adaptativa
- El umbral varía según la región local
- Maneja variaciones de iluminación
- Dos variantes: media y gaussiana

#### Binarización Manual
- Usa un umbral fijo definido por el usuario
- Control total sobre la segmentación
- Útil cuando se conoce el rango de valores

### 5. **Detección de Bordes (Canny)**
- Detector de bordes de alta calidad
- Reduce ruido antes de detectar bordes
- Encuentra bordes finos y continuos
- Configurable con umbrales de hysteresis

## 📊 Configuración del Sistema

El archivo `config.py` centraliza toda la configuración:

```python
# Configuración de procesamiento
PROCESAMIENTO = {
    'tamaño_estandar': (512, 512),
    'filtro_gaussiano': {
        'kernel_size': 5,
        'sigma': 1.0
    },
    'canny': {
        'umbral_bajo': 50,
        'umbral_alto': 150
    }
}
```

## 🧪 Pruebas y Ejemplos

### Ejecutar Todos los Ejemplos
```bash
python ejemplos_uso.py
```

### Crear Imagen de Prueba
Si no tienes imágenes disponibles, el sistema puede crear una imagen sintética:
```python
crear_imagen_de_prueba()
```

### Comparar Métodos
```python
ejemplo_comparacion_filtros()    # Compara diferentes filtros
ejemplo_binarizacion()           # Compara métodos de binarización
ejemplo_deteccion_bordes()       # Prueba detección de bordes
```

## 📁 Datos de Entrada

### Formatos Soportados
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`, `.tif`

### Organización de Datos
```
data/imagenes_originales/
├── tornillos/
│   ├── tornillo_01.jpg
│   ├── tornillo_02.png
│   └── ...
├── tuercas/
├── arandelas/
└── clavos/
```

## 🔬 Metodología

El proyecto sigue una metodología clásica de visión artificial:

1. **Adquisición**: Carga de imágenes desde archivos
2. **Preprocesamiento**: Mejora de calidad y estandarización
3. **Segmentación**: Separación de objetos del fondo
4. **Extracción de Características**: Cálculo de descriptores geométricos
5. **Clasificación**: Asignación a categorías basada en características

## 🎯 Próximos Pasos

### Módulos en Desarrollo

1. **Extracción de Características** (`extraccion_caracteristicas.py`)
   - Análisis de contornos
   - Cálculo de momentos geométricos
   - Descriptores de forma
   - Análisis de textura

2. **Clasificador** (`clasificador.py`)
   - Clasificador basado en reglas
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Evaluación de rendimiento

3. **Interfaz de Usuario**
   - GUI para carga de imágenes
   - Visualización interactiva
   - Configuración de parámetros

## 📈 Resultados Esperados

El sistema debería ser capaz de:
- Procesar imágenes con diferentes condiciones de iluminación
- Identificar correctamente elementos de ferretería
- Manejar variaciones en orientación y escala
- Proporcionar confianza en las clasificaciones

## 🤝 Contribución

Este proyecto es parte del trabajo final de la materia de Inteligencia Artificial. Las contribuciones siguientes están planificadas:

1. Mejoras en el preprocesamiento
2. Implementación de nuevos descriptores
3. Optimización de algoritmos
4. Ampliación del dataset

## 📝 Documentación Adicional

- [`adquisicion_datos.py`](src/adquisicion_datos.py): Documentación detallada del módulo de preprocesamiento
- [`config.py`](config.py): Configuración completa del sistema
- [`ejemplos_uso.py`](ejemplos_uso.py): Ejemplos prácticos de uso

## 🎓 Contexto Académico

Este proyecto se desarrolla como parte del curso de Inteligencia Artificial, enfocándose en:
- Técnicas clásicas de procesamiento de imágenes
- Algoritmos fundamentales de visión artificial
- Metodologías de desarrollo de sistemas inteligentes
- Aplicación práctica de conceptos teóricos

## 📞 Soporte

Para preguntas o problemas:
1. Revisa la documentación en los archivos fuente
2. Ejecuta `python config.py` para verificar la configuración
3. Prueba los ejemplos en `ejemplos_uso.py`

---

**Proyecto Final - Inteligencia Artificial**  
**Fecha**: Octubre 2025  
**Objetivo**: Sistema de Reconocimiento de Elementos de Ferretería

## Características Principales

### 🔍 **Pipeline Completo de Visión Artificial**
- **Adquisición**: Captura desde cámara web o archivos de imagen
- **Preprocesado**: Reajuste de iluminación, escala de grises, filtrado y binarización
- **Segmentación**: Detección de contornos y aproximación polinomial
- **Extracción**: Momentos invariantes de Hu y características geométricas
- **Clasificación**: Algoritmos KNN y K-Means implementados desde cero
- **Interpretación**: Generación de reportes y visualizaciones

### 🎤 **Reconocimiento de Voz**
- Comandos en español
- Procesamiento en tiempo real
- Respuestas por síntesis de voz

### 🤖 **Algoritmos Implementados**
- **KNN propio**: Con múltiples métricas de distancia
- **K-Means propio**: Con inicialización K-Means++
- **Momentos de Hu**: Para invarianza a transformaciones
- **Características geométricas**: Área, perímetro, circularidad, etc.

### 📊 **Análisis y Reportes**
- Estadísticas detalladas
- Gráficos automatizados
- Reportes en JSON y texto
- Visualizaciones interactivas

## Instalación

### Prerrequisitos
- Python 3.7+
- Cámara web (opcional)
- Micrófono (opcional para comandos de voz)

### Instalación Automática
```bash
# Ejecutar configuración automática
python setup.py
```

### Instalación Manual
```bash
# Instalar dependencias
pip install -r requirements.txt

# Crear directorios necesarios
mkdir -p datos/imagenes_entrenamiento datos/imagenes_prueba modelos resultados
```

## Uso Rápido

### Ejecución Básica
```bash
# Ejecutar con interfaz completa
python main.py

# Procesar imagen específica
python main.py --imagen datos/test.jpg

# Ejecutar sin reconocimiento de voz
python main.py --sin-voz

# Modo entrenamiento
python main.py --entrenar
```

### Comandos de Voz Disponibles
- **"visión artificial captura imagen"** - Capturar nueva imagen
- **"visión artificial procesa imagen"** - Procesar imagen actual
- **"visión artificial muestra resultados"** - Mostrar resultados
- **"visión artificial cuántos hay"** - Contar elementos detectados
- **"visión artificial busca tuercas"** - Buscar elementos específicos

## Estructura del Proyecto

```
ProyectoFinalIA/
├── src/                           # Código fuente
│   ├── base.py                    # Clases base y arquitectura
│   ├── adquisicion.py             # Adquisición de imágenes
│   ├── preprocesado.py            # Preprocesado de imágenes
│   ├── segmentacion.py            # Segmentación y contornos
│   ├── extraccion_caracteristicas.py  # Extracción de características
│   ├── algoritmos_clasificacion.py    # KNN y K-Means implementados
│   ├── clasificacion.py          # Clasificador de elementos
│   ├── reconocimiento_voz.py      # Reconocimiento de voz
│   ├── interpretacion.py          # Interpretación y reportes
│   └── sistema_principal.py       # Sistema integrado
├── datos/                         # Datos de entrada
├── modelos/                       # Modelos entrenados
├── resultados/                    # Resultados de análisis
├── tests/                         # Tests unitarios
├── ejemplos/                      # Ejemplos de uso
└── main.py                       # Archivo principal
```

## Pipeline de Procesamiento

1. **Adquisición** → Captura imagen desde cámara o archivo
2. **Preprocesado** → Mejora imagen (gamma, CLAHE, filtros, binarización)
3. **Segmentación** → Detecta contornos y calcula aproximaciones polinomiales
4. **Extracción** → Calcula momentos de Hu y características geométricas
5. **Clasificación** → Aplica KNN y K-Means para identificar elementos
6. **Interpretación** → Genera reportes, estadísticas y visualizaciones

## Algoritmos Implementados

### K-Nearest Neighbors (KNN)
- **Métricas de distancia**: Euclidiana, Manhattan, Coseno
- **Tipos de pesos**: Uniforme, Ponderado por distancia
- **Validación cruzada** integrada

### K-Means
- **Inicialización**: Aleatoria o K-Means++
- **Múltiples métricas** de distancia
- **Criterios de convergencia** configurables

### Momentos Invariantes de Hu
- **7 momentos** calculados automáticamente
- **Invariantes** a traslación, rotación y escala
- **Normalización** logarítmica aplicada

## Ejemplo de Uso Programático

```python
from src.sistema_principal import SistemaReconocimientoElementosFerreteros

# Configurar sistema
config = {
    'usar_camara': True,
    'habilitar_voz': True,
    'guardar_resultados': True
}

# Inicializar
sistema = SistemaReconocimientoElementosFerreteros(config)

# Capturar y procesar
sistema.capturar_imagen()
sistema.procesar_imagen_actual()

# Ver resultados
print(f"Elementos detectados: {sistema.contar_elementos()}")
sistema.mostrar_resultados()
```

## Tests

```bash
# Ejecutar todos los tests
python tests/test_sistema.py

# Ejecutar ejemplos
python ejemplos/ejemplo_uso.py
```

## Autor

**Proyecto Final de Inteligencia Artificial - Visión Artificial**  
**Año**: 2025