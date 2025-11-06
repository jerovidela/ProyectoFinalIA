import os
import json
import time
import pickle
import tempfile
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .binary import process_images
from .image_params import extract_image_features
from .audio_params import extract_and_save_features, compute_features_exact
from .my_kmeans import kmeans_from_json
from .my_knn import knn_from_featurejson
from .bayes import proporcion, contar


class ModelManager:
    def __init__(self):
        self.kmeans_model = None
        self.knn_model = None
        self.audio_stats = None
        self.classification_results = []
        
    def train_models(self):
        print("🚀 Iniciando entrenamiento de modelos...")
        print("📸 Procesando imágenes...")
        process_images("img2", "out")
        extract_image_features("out", "outjson")
        
        print("🔍 Entrenando modelo KMeans...")
        self.kmeans_model = kmeans_from_json(
            json_dir="outjson",
            param_keys=["circle_area_ratio", "hu_moment_1", "angles_min", "hu_moment_2", "curvature_max"],
            n_clusters=4,
            n_init=10,
            max_iterations=300,
            tolerance=1e-4,
            random_seed=42,
            verbose=False
        )

        print("🎵 Procesando audios...")
        records, self.audio_stats = extract_and_save_features("audio", "features_json", "features_json_data")
        
        print("🎯 Entrenando modelo KNN...")
        self.knn_model = knn_from_featurejson(
            json_dir="features_json",
            feature_keys=("zcr_std_z", "rolloff95_std_z", "mfcc_std_4_z"),
            is_classification=True,
            n_neighbors=5,
            weights="distance",
            metric="manhattan",
            p=2,
            standardize=True,
            test_fraction=0.3,
            random_seed=42,
            max_examples=10,
            verbose=False
        )
        
        # Estandarizar la estructura del modelo recién entrenado
        self._standardize_kmeans_model()
        
        self.save_models()
        print("✅ Entrenamiento completado y modelos guardados!")
        print(f"📊 Precisión KNN: {self.knn_model['metric_value']:.4f}")
        
    def _create_cluster_mapping(self):
        """Crea el mapeo cluster -> clase basado en los datos de entrenamiento"""
        if not self.kmeans_model:
            return {}
            
        try:
            from collections import Counter
            
            # Obtener asignaciones de cluster y etiquetas verdaderas
            assignments = self.kmeans_model['assignments']
            true_labels = self.kmeans_model['true_labels']
            
            # Para cada cluster, encontrar la clase más frecuente
            cluster_to_class = {}
            n_clusters = len(self.kmeans_model['centroids'])
            
            for cluster_id in range(n_clusters):
                # Obtener etiquetas de todas las muestras asignadas a este cluster
                cluster_mask = assignments == cluster_id
                cluster_labels = [true_labels[i] for i, mask in enumerate(cluster_mask) if mask]
                
                if cluster_labels:
                    # Encontrar la clase más frecuente en este cluster
                    label_counts = Counter(cluster_labels)
                    most_common_label = label_counts.most_common(1)[0][0]
                    # Normalizar a minúsculas para compatibilidad con módulo bayesiano
                    cluster_to_class[cluster_id] = most_common_label.lower()
                    
                    print(f"🎯 Cluster {cluster_id} → {most_common_label.lower()} ({len(cluster_labels)} muestras)")
                else:
                    cluster_to_class[cluster_id] = "desconocido"
                    print(f"⚠️ Cluster {cluster_id} → sin muestras asignadas")
            
            return cluster_to_class
            
        except Exception as e:
            print(f"❌ Error creando mapeo de clusters: {e}")
            # Fallback al mapeo fijo si hay problemas
            return {0: "tornillo", 1: "clavo", 2: "arandela", 3: "tuerca"}
    
    def _standardize_kmeans_model(self):
        """Estandariza la estructura del modelo KMeans recién entrenado"""
        if not self.kmeans_model:
            return
            
        # Crear mapeo cluster -> clase basado en los datos de entrenamiento
        cluster_to_class = self._create_cluster_mapping()
        
        # Crear la estructura estándar que esperan los métodos de predicción
        standardized_model = {
            'centroids': self.kmeans_model.get('centroids'),
            'scaler': self.kmeans_model.get('scaler'),
            'param_keys': ["circle_area_ratio", "hu_moment_1", "angles_min", "hu_moment_2", "curvature_max"],
            'cluster_to_class': cluster_to_class
        }
        
        # Reemplazar el modelo con la estructura estándar
        self.kmeans_model = standardized_model
        print("🔄 Estructura del modelo KMeans estandarizada")
        
    def save_models(self):
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # El modelo ya está estandarizado, solo guardarlo
        with open(models_dir / "kmeans_model.pkl", "wb") as f:
            pickle.dump(self.kmeans_model, f)
            
        knn_data = {
            'X_train': self.knn_model['X_train'],
            'y_train': self.knn_model['y_train'],
            'mu': self.knn_model['mu'],
            'sigma': self.knn_model['sigma'],
            'class_to_int': self.knn_model['class_to_int'],
            'int_to_class': self.knn_model['int_to_class'],
            'feature_keys': ("zcr_std_z", "rolloff95_std_z", "mfcc_std_4_z")
        }
        with open(models_dir / "knn_model.pkl", "wb") as f:
            pickle.dump(knn_data, f)
            
        with open(models_dir / "audio_stats.json", "w") as f:
            json.dump(self.audio_stats, f, indent=2)
    
    def load_models(self):
        models_dir = Path("models")
        
        try:
            with open(models_dir / "kmeans_model.pkl", "rb") as f:
                self.kmeans_model = pickle.load(f)
                
            with open(models_dir / "knn_model.pkl", "rb") as f:
                self.knn_model = pickle.load(f)
                
            with open(models_dir / "audio_stats.json", "r") as f:
                self.audio_stats = json.load(f)
                
            print("✅ Modelos cargados exitosamente!")
            return True
        except FileNotFoundError:
            print("⚠️ No se encontraron modelos guardados. Se requiere entrenamiento.")
            return False
    
    def predict_image(self, image_path):
        if not self.kmeans_model:
            print("❌ Modelo KMeans no está cargado")
            return None
            
        try:
            temp_dir = Path("temp_img")
            temp_dir.mkdir(exist_ok=True)
            
            import shutil
            temp_img = temp_dir / Path(image_path).name
            shutil.copy2(image_path, temp_img)
            
            print(f"🔄 Procesando imagen: {Path(image_path).name}")
            
            process_images(str(temp_dir), "temp_out")
            extract_image_features("temp_out", "temp_json")
            
            json_file = Path("temp_json") / f"{Path(image_path).stem}.json"
            if not json_file.exists():
                print(f"❌ No se pudo generar archivo de características para {Path(image_path).name}")
                return None
                
            with open(json_file, "r") as f:
                features = json.load(f)
            
            # Extraer vector de características
            param_keys = self.kmeans_model['param_keys']
            feature_vector = [features[key] for key in param_keys]
            
            # Estandarizar
            import numpy as np
            X = np.array([feature_vector])
            X_scaled = self.kmeans_model['scaler'].transform(X)
            
            # Predecir cluster
            centroids = self.kmeans_model['centroids']
            distances = np.sum((X_scaled - centroids) ** 2, axis=1)
            cluster = int(np.argmin(distances))
            
            # Usar mapeo dinámico cluster -> clase basado en entrenamiento
            cluster_to_class = self.kmeans_model.get('cluster_to_class', {})
            if not cluster_to_class:
                # Fallback si no hay mapeo guardado
                cluster_to_class = {0: "tornillo", 1: "clavo", 2: "arandela", 3: "tuerca"}
                print("⚠️ Usando mapeo por defecto - se recomienda reentrenar el modelo")
                
            predicted_class = cluster_to_class.get(cluster, "desconocido")
            
            print(f"✅ Imagen clasificada como: {predicted_class} (cluster {cluster})")
            
            return predicted_class
            
        except Exception as e:
            print(f"❌ Error en predicción de imagen: {e}")
            return None
        finally:
            # Limpiar archivos temporales
            import shutil
            for temp_folder in ["temp_img", "temp_out", "temp_json"]:
                if Path(temp_folder).exists():
                    shutil.rmtree(temp_folder, ignore_errors=True)
    
    def predict_audio(self, audio_path):
        """Predice la clase de un audio usando KNN"""
        if not self.knn_model or not self.audio_stats:
            print("❌ Modelo KNN o estadísticas de audio no están cargados")
            return None
            
        try:
            print(f"🔄 Procesando audio: {Path(audio_path).name}")
            
            # Extraer características del audio
            zcr_std, rolloff95_std, mfcc_std_4 = compute_features_exact(audio_path)
            
            # Estandarizar usando estadísticas del entrenamiento
            import numpy as np
            stats = self.audio_stats['z_score_stats']
            
            zcr_std_z = (zcr_std - stats['zcr_std']['mean']) / stats['zcr_std']['std']
            rolloff95_std_z = (rolloff95_std - stats['rolloff95_std']['mean']) / stats['rolloff95_std']['std']
            mfcc_std_4_z = (mfcc_std_4 - stats['mfcc_std_4']['mean']) / stats['mfcc_std_4']['std']
            
            # Vector de características
            X_new = np.array([[zcr_std_z, rolloff95_std_z, mfcc_std_4_z]])
            
            # Aplicar estandarización del modelo
            X_new = (X_new - self.knn_model['mu']) / self.knn_model['sigma']
            
            # Calcular distancias a datos de entrenamiento
            X_train = self.knn_model['X_train']
            distances = np.sum(np.abs(X_new - X_train), axis=1)  # Manhattan
            
            # Obtener k vecinos más cercanos
            k = 5
            knn_indices = np.argsort(distances)[:k]
            knn_distances = distances[knn_indices]
            knn_labels = self.knn_model['y_train'][knn_indices]
            
            # Pesos por distancia
            weights = 1.0 / np.maximum(knn_distances, 1e-12)
            
            # Votar con pesos
            unique_labels = np.unique(self.knn_model['y_train'])
            scores = np.zeros(len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = knn_labels == label
                scores[i] = np.sum(weights[mask])
            
            predicted_label_int = unique_labels[np.argmax(scores)]
            predicted_class = self.knn_model['int_to_class'][predicted_label_int]
            
            print(f"✅ Audio clasificado como: {predicted_class}")
            
            return predicted_class
            
        except Exception as e:
            print(f"❌ Error en predicción de audio: {e}")
            return None
    
    def add_classification_result(self, filename, predicted_class, file_type):
        """Agrega un resultado de clasificación"""
        self.classification_results.append({
            'filename': filename,
            'class': predicted_class,
            'type': file_type,
            'timestamp': time.time()
        })
        
        # Mantener solo las últimas 50 clasificaciones
        if len(self.classification_results) > 50:
            self.classification_results = self.classification_results[-50:]
    
    def get_sample_labels(self, count=10):
        """Obtiene las últimas clasificaciones como muestra para Bayes"""
        if len(self.classification_results) < count:
            # Si no hay suficientes, usar todas las disponibles
            recent = self.classification_results
        else:
            recent = self.classification_results[-count:]
        
        return [r['class'] for r in recent if r['type'] == 'image']


class FileMonitor(FileSystemEventHandler):
    """Monitorea cambios en las carpetas de comando e imágenes"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        filename = os.path.basename(file_path)
        
        print(f"🔍 Archivo detectado: {file_path}")
        
        # Esperar un poco para asegurar que el archivo esté completamente escrito
        time.sleep(1.0)
        
        try:
            if "comando" in file_path and filename.lower().endswith('.wav'):
                print(f"🎵 Procesando comando de audio: {filename}")
                self.handle_audio_command(file_path)
            elif "img_new" in file_path and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"📸 Procesando imagen: {filename}")
                self.handle_new_image(file_path)
            else:
                print(f"⚠️ Archivo ignorado (tipo no soportado): {filename}")
        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
    
    # def on_modified(self, event):
    #     # También detectar modificaciones (por si acaso)
    #     if not event.is_directory:
    #         self.on_created(event)
    
    def handle_new_image(self, image_path):
        """Procesa una nueva imagen"""
        print(f"🖼️ Nueva imagen detectada: {os.path.basename(image_path)}")
        
        try:
            predicted_class = self.model_manager.predict_image(image_path)
            if predicted_class:
                print(f"🔍 Clasificación: {predicted_class}")
                self.model_manager.add_classification_result(
                    os.path.basename(image_path), 
                    predicted_class, 
                    'image'
                )
            else:
                print("❌ Error al clasificar imagen")
        except Exception as e:
            print(f"❌ Error procesando imagen: {e}")
    
    def handle_audio_command(self, audio_path):
        """Procesa un comando de audio"""
        print(f"🎵 Comando de audio detectado: {os.path.basename(audio_path)}")
        
        try:
            predicted_command = self.model_manager.predict_audio(audio_path)
            if predicted_command:
                print(f"🎯 Comando reconocido: {predicted_command}")
                result = self.execute_command(predicted_command)
                self.model_manager.add_classification_result(
                    os.path.basename(audio_path), 
                    predicted_command, 
                    'audio'
                )
                if result == 'exit':
                    print("🛑 Comando de salida recibido")
                    os._exit(0)
            else:
                print("❌ Error al reconocer comando")
        except Exception as e:
            print(f"❌ Error procesando audio: {e}")
    
    def execute_command(self, command):
        """Ejecuta la acción correspondiente al comando"""
        # Obtener muestra de clasificaciones recientes
        sample_labels = self.model_manager.get_sample_labels(10)
        
        if not sample_labels:
            print("⚠️ No hay clasificaciones de imágenes disponibles para análisis bayesiano")
            return
        
        # Normalizar etiquetas a minúsculas para el módulo bayesiano
        normalized_labels = [label.lower() for label in sample_labels]
        
        print(f"📊 Muestra actual: {sample_labels}")
        
        try:
            if command.lower() == 'proporcion':
                props = proporcion(normalized_labels)
                print("\n📈 PROPORCIONES ESTIMADAS:")
                print("=" * 40)
                for clase, prop in props.items():
                    print(f"  {clase:12s}: {prop:6.2%}")
                    
            elif command.lower() == 'contar':
                counts = contar(normalized_labels, 1000)
                print("\n🔢 CONTEO ESTIMADO (de 1000 piezas):")
                print("=" * 40)
                for clase, count in counts.items():
                    print(f"  {clase:12s}: {count:4d} piezas")
                    
            elif command.lower() == 'salir':
                print("\n👋 Finalizando aplicación...")
                return 'exit'
            else:
                print(f"⚠️ Comando no reconocido: {command}")
                
        except Exception as e:
            print(f"❌ Error en análisis bayesiano: {e}")
            print("💡 Verificando compatibilidad de etiquetas...")
            print(f"   Etiquetas recibidas: {sample_labels}")
            print(f"   Etiquetas normalizadas: {normalized_labels}")
        
        print()


def start_monitoring():
    """Función principal que inicia todo el sistema"""
    # Crear directorios necesarios
    Path("comando").mkdir(exist_ok=True)
    Path("img_new").mkdir(exist_ok=True)
    
    # Inicializar gestor de modelos
    model_manager = ModelManager()
    
    # Cargar o entrenar modelos
    if not model_manager.load_models():
        model_manager.train_models()
    
    # Procesar archivos existentes en las carpetas monitoreadas
    print("\n🔍 Procesando archivos existentes...")
    
    # Configurar monitor
    event_handler = FileMonitor(model_manager)
    
    # Procesar imágenes existentes en img_new/
    img_new_dir = Path("img_new")
    for img_file in img_new_dir.glob("*.jpg"):
        print(f"📸 Procesando imagen existente: {img_file.name}")
        try:
            event_handler.handle_new_image(str(img_file))
        except Exception as e:
            print(f"❌ Error procesando {img_file.name}: {e}")

    
    # Procesar audios existentes en comando/
    comando_dir = Path("comando")
    for audio_file in comando_dir.glob("*.wav"):
        print(f"🎵 Procesando audio existente: {audio_file.name}")
        try:
            event_handler.handle_audio_command(str(audio_file))
        except Exception as e:
            print(f"❌ Error procesando {audio_file.name}: {e}")
    
    # Configurar monitoreo de archivos para nuevos archivos
    observer = Observer()
    observer.schedule(event_handler, "comando", recursive=False)
    observer.schedule(event_handler, "img_new", recursive=False)
    
    print("\n🚀 Sistema de monitoreo iniciado!")
    print("📁 Monitoreando:")
    print("   • comando/ - para archivos de audio (.wav)")
    print("   • img_new/ - para imágenes (.jpg, .jpeg, .png)")
    print("\n💡 Comandos disponibles:")
    print("   • 'proporcion' - muestra proporciones estimadas")
    print("   • 'contar' - muestra conteo estimado de 1000 piezas")
    print("   • 'salir' - finaliza la aplicación")
    print("\n⏳ Esperando archivos nuevos... (Ctrl+C para salir)")
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Deteniendo monitoreo...")
        observer.stop()
    
    observer.join()
    print("✅ Sistema finalizado.")


if __name__ == "__main__":
    start_monitoring()