import os
import json
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PARAM_KEYS = [
    "circle_area_ratio",
    "hu_moment_1", 
    "angles_min",
    "hu_moment_2",
    "curvature_max",
]

json_dir = "outjson"
pattern = os.path.join(json_dir, "*.json")
files = sorted(glob.glob(pattern))

filenames = []
labels = []
params_list = []

for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except:
        continue
        
    if "filename" in obj:
        filename = obj["filename"]
    else:
        filename = os.path.basename(fp)
    
    base = os.path.basename(filename)
    base_no_ext = os.path.splitext(base)[0]
    label = "".join(ch for ch in base_no_ext if not ch.isdigit())
    label = label.strip('_-. ').capitalize()
    
    params = []
    valid = True
    for k in PARAM_KEYS:
        if k in obj and obj[k] is not None:
            try:
                params.append(float(obj[k]))
            except:
                valid = False
                break
        else:
            valid = False
            break
    
    if valid:
        filenames.append(filename)
        labels.append(label)
        params_list.append(params)

print(f"Datos cargados: {len(filenames)} elementos")

params_matrix = np.array(params_list)
scaler = StandardScaler()
normalized_params = scaler.fit_transform(params_matrix)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(normalized_params)

print(f"\nClustering completado con 4 clusters")

# Mostrar composición de cada cluster
print("\n" + "="*50)
print("COMPOSICIÓN DE CLUSTERS")
print("="*50)

for cluster_id in range(4):
    mask = cluster_labels == cluster_id
    cluster_files = [filenames[i] for i in range(len(filenames)) if mask[i]]
    cluster_labels_list = [labels[i] for i in range(len(labels)) if mask[i]]
    
    print(f"\nCLUSTER {cluster_id} ({len(cluster_files)} elementos):")
    print("-" * 30)
    
    for i, (file, label) in enumerate(zip(cluster_files, cluster_labels_list)):
        print(f"  {i+1:2d}. {file} ({label})")
    
    from collections import Counter
    type_counts = Counter(cluster_labels_list)
    print(f"\nResumen por tipo:")
    for tipo, count in sorted(type_counts.items()):
        percentage = (count / len(cluster_files)) * 100
        print(f"    {tipo}: {count} elementos ({percentage:.1f}%)")
