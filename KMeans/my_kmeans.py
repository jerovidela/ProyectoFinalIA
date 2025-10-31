import os
import json
import glob
import numpy as np
import random

PARAM_KEYS = [
    "circle_area_ratio",
    "hu_moment_1", 
    "angles_min",
    "hu_moment_2",
    "curvature_max",
]

n_clusters = 4
max_iterations = 100
tolerance = 1e-4
random_seed = 42

np.random.seed(random_seed)
random.seed(random_seed)

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

data_matrix = np.array(params_list)
n_samples, n_features = data_matrix.shape

means = np.mean(data_matrix, axis=0)
stds = np.std(data_matrix, axis=0, ddof=0)
normalized_data = (data_matrix - means) / stds

best_inertia = float('inf')
best_centroids = None
best_labels = None
n_init = 10

for init_run in range(n_init):
    
    centroids = np.zeros((n_clusters, n_features))

    centroids[0] = normalized_data[np.random.randint(n_samples)]
    
    for c in range(1, n_clusters):
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist = float('inf')
            for j in range(c):
                dist = np.sum((normalized_data[i] - centroids[j]) ** 2)
                if dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist
        probabilities = distances / np.sum(distances)
        cumulative_prob = np.cumsum(probabilities)
        r = np.random.random()
        for i in range(n_samples):
            if r <= cumulative_prob[i]:
                centroids[c] = normalized_data[i]
                break

cluster_assignments = np.zeros(n_samples, dtype=int)

for iteration in range(max_iterations):
    
    for i in range(n_samples):
        min_distance = float('inf')
        best_cluster = 0
        for j in range(n_clusters):
            distance = np.sum((normalized_data[i] - centroids[j]) ** 2)
            if distance < min_distance:
                min_distance = distance
                best_cluster = j
        cluster_assignments[i] = best_cluster
    
    new_centroids = np.zeros((n_clusters, n_features))
    for j in range(n_clusters):
        cluster_points = normalized_data[cluster_assignments == j]
        if len(cluster_points) > 0:
            new_centroids[j] = np.mean(cluster_points, axis=0)
        else:
            new_centroids[j] = centroids[j]
    
    centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
    
    centroids = new_centroids
    
    if centroid_shift < tolerance:
        break

current_inertia = 0
for i in range(n_samples):
    cluster_id = cluster_assignments[i]
    distance_squared = np.sum((normalized_data[i] - centroids[cluster_id]) ** 2)
    current_inertia += distance_squared

if current_inertia < best_inertia:
    best_inertia = current_inertia
    best_centroids = np.copy(centroids)
    best_labels = np.copy(cluster_assignments)

centroids = best_centroids
cluster_assignments = best_labels

print("\n" + "="*50)
print("COMPOSICIÓN DE CLUSTERS")
print("="*50)

for cluster_id in range(n_clusters):
    mask = cluster_assignments == cluster_id
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