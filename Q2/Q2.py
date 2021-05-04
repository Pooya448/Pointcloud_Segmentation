import numpy as np
import open3d as o3d
import os
import glob
import random

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering

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
    neighbors = NearestNeighbors(n_neighbors=K+1, algorithm='auto').fit(points)
    distances, indices = neighbors.kneighbors(points)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    adjacency_matrix = neighbors.kneighbors_graph(points).toarray()
    np.fill_diagonal(adjacency_matrix, 0)
    print(f"KNN -> K = {K}")
    return adjacency_matrix, indices, distances

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def compute_angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def compute_weights(adj, points, boundary_probs, option):
    adj_weights = np.zeros_like(adj)

    rows, cols = np.where(adjacency_matrix==1)
    coordinates = zip(rows, cols)

    if option == 'angle':
        for i, j in coordinates:
            v1 = points[i, :]
            v2 = points[j, :]
            adj_weights[i, j] = np.rad2deg(compute_angle(v1, v2))

    elif option == 'prob':
        for i, j in coordinates:
            adj_weights[i, j] = np.maximum(boundary_probs[i], boundary_probs[j])

    if not check_symmetric(adj_weights):
        adj_weights = (adj_weights + adj_weights.T) / 2

    print(f"Computing weights using -> {option}")
    return adj_weights

def cluster(w, N):
    print(f"Clustering for -> {N} segments")
    clustering = SpectralClustering(n_clusters=N, assign_labels='kmeans', affinity='precomputed', random_state=0, verbose=True).fit(w)
    regions = clustering.labels_
    regions = regions.reshape(-1, 1)
    return regions


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

res_path = './Results/'
if os.path.exists(res_path):
    for f in glob.glob('./Results/*.*'):
        os.remove(f)
else:
    os.mkdir(res_path)

pcd = o3d.io.read_point_cloud("points.xyzn")
print(pcd)

points, probs, labels = setup()

K = 5
adjacency_matrix, indices, distances = KNN(points, K)

option = 'angle'
weights = compute_weights(adjacency_matrix, points, probs, option)

N = 2
regions = cluster(weights, N)

palette = generate_palette(N)
region_colors = []
for i in regions.tolist():
    region_colors.append(palette[i, :])
region_colors = np.array(region_colors).reshape(10000, 3)

pcd.colors = o3d.utility.Vector3dVector(region_colors)
pcd.estimate_normals()
o3d.io.write_point_cloud(res_path + f"segmented_pointcloud_{option}.ply", pcd)
radii = [0.05, 0.1, 0.2, 0.4]
radii = o3d.utility.DoubleVector(radii)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
o3d.io.write_triangle_mesh(res_path + f"segmented_mesh_{option}.ply", mesh)

option = 'prob'
weights = compute_weights(adjacency_matrix, points, probs, option)

regions = cluster(weights, N)

region_colors = []
for i in regions.tolist():
    region_colors.append(palette[i, :])
region_colors = np.array(region_colors).reshape(10000, 3)

pcd.colors = o3d.utility.Vector3dVector(region_colors)
pcd.estimate_normals()
o3d.io.write_point_cloud(res_path + f"segmented_pointcloud_{option}.ply", pcd)
radii = [0.05, 0.1, 0.2, 0.4]
radii = o3d.utility.DoubleVector(radii)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
o3d.io.write_triangle_mesh(res_path + f"segmented_mesh_{option}.ply", mesh)
