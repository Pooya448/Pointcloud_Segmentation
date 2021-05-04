import numpy as np
import open3d as o3d
import os
import glob
import random
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

def generate_palette(n):
    if n > 7:
        return np.random.uniform(size=(n, 3))
    else:
        colors = [
            [0, 0, 128],
            [128, 0, 0],
            [250, 190, 212],
            [255, 255, 25],
            [70, 240, 240],
            [240, 50, 230],
            [0, 0, 0],
        ]
        random.shuffle(colors)
        return np.array(colors) / 255

def compute_f1(probs, labels):
    pred_softmax = softmax(probs)

    threshold = 2 * np.mean(pred_softmax)
    print(f"Threshold value: {threshold}")
    inferred = np.where(pred_softmax > threshold, 1, 0)

    f1 = f1_score(labels, inferred)

    inferred = inferred.reshape((-1, 1))
    return f1, inferred

def setup():
    points = []
    points_normal = []
    labels = []
    probs = []

    with open('./example.pts', 'r') as f:
        for _ in range(10000):

            line = f.readline().strip().split(' ')
            xyz = line[:3]
            nxnynz = line[3:6]
            label = line[-2]
            prob = line[-1]

            points.append(xyz)
            points_normal.append(nxnynz)
            labels.append(label)
            probs.append(prob)

    probs = np.array(probs).astype(np.float32)
    labels = np.array(labels).astype(np.int8)
    points = np.array(points).astype(np.float32)

    return points, probs, labels

def KNN(points, K):
    neighbors = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree').fit(points)
    distances, indices = neighbors.kneighbors(points)
    adjacency_matrix = neighbors.kneighbors_graph(points).toarray()
    print(f"KNN -> K = {K}")
    return indices[:, 1:]

def contains_boundary(inferred, vs):
    for v in vs:
        if inferred[v] == 1:
            return True
    return False

def watershed_segmentation(points, knn, inferred):
    region_counter = 1
    segment_labels = np.repeat(-1, 10000)

    nodes = range(10000)
    visited = []
    queue = []

    while len(visited) < 10000:

        not_visited = set(nodes) - set(visited)
        seed = random.sample(not_visited, 1)[0]

        visited.append(seed)
        if inferred[seed] == 1:
            segment_labels[seed] = 0
        else:
            region_counter += 1
            segment_labels[seed] = region_counter
            queue.append(seed)

            while queue:
                s = queue.pop(0)

                neighbors = knn[s]
                check = contains_boundary(inferred, neighbors)

                for v in knn[s]:
                    if v not in visited:
                        visited.append(v)
                        if inferred[v] == 0:
                            if not check:
                                queue.append(v)
                            segment_labels[v] = region_counter
                        else:
                            segment_labels[v] = 0

    region_count = region_counter + 1
    print(f"Total number of parts -> {region_counter}")

    assert np.count_nonzero(segment_labels == 0) == np.count_nonzero(inferred == 1)
    assert np.count_nonzero(segment_labels == -1) == 0

    return segment_labels.astype(np.uint8), region_count

res_path = './Results/'
if os.path.exists(res_path):
    for f in glob.glob('./Results/*.*'):
        os.remove(f)
else:
    os.mkdir(res_path)

pcd = o3d.io.read_point_cloud("points.xyzn")
print(pcd)

points, probs, labels = setup()
f1, inferred = compute_f1(probs, labels)

init_colors = [[0, 160/255, 217/255] if x == 0 else [1, 0, 0] for x in inferred]
init_colors = np.array(init_colors)
pcd.colors = o3d.utility.Vector3dVector(init_colors)
pcd.estimate_normals()
o3d.io.write_point_cloud(res_path + "initial_pointcloud.ply", pcd)

radii = [0.05, 0.1, 0.2, 0.4]
radii = o3d.utility.DoubleVector(radii)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
o3d.io.write_triangle_mesh(res_path + "initial_mesh.ply", mesh)

print(f"F1 Score -> {f1*100}")

K = 5
knn = KNN(points, K)
regions, region_count = watershed_segmentation(points, knn, inferred)

palette = generate_palette(region_count)
region_colors = []
for i in regions.tolist():
    region_colors.append(palette[i])
region_colors = np.array(region_colors)

pcd.colors = o3d.utility.Vector3dVector(region_colors)
pcd.estimate_normals()
o3d.io.write_point_cloud(res_path + "segmented_pointcloud.ply", pcd)

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
o3d.io.write_triangle_mesh(res_path + "segmented_mesh.ply", mesh)
