import numpy as np
from pymilvus import connections, Collection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
from tqdm import tqdm  # For progress bar
import cupy as cp

# Milvus configuration
MILVUS_CONFIG = {
    "host": "127.0.0.1",
    "port": "19530",
    "collection_name": "embedded_texts",
    "embedding_dim": 1024,
}

def connect_to_milvus(config):
    """Connect to Milvus and return the collection."""
    connections.connect(host=config["host"], port=config["port"])
    print("Connected to Milvus")
    collection = Collection(config["collection_name"])
    return collection

def fetch_vectors(collection, limit=None, batch_size=5000):
    """Fetch vectors from the Milvus collection in batches."""
    vectors = []
    max_query_window = 16384  # Milvus' maximum query result window
    if limit and limit > max_query_window:
        print(f"Reducing the fetch limit from {limit} to {max_query_window} due to Milvus constraints.")
        limit = max_query_window

    total_entities = collection.num_entities
    fetch_limit = limit or total_entities

    num_batches = (fetch_limit + batch_size - 1) // batch_size

    print("Fetching vectors with progress bar:")
    for i in tqdm(range(num_batches), desc="Loading Data"):
        start = i * batch_size
        end = min(start + batch_size, fetch_limit)
        try:
            query_result = collection.query(
                expr="",
                output_fields=["embedding"],
                offset=start,
                limit=end - start
            )
            vectors.extend(item["embedding"] for item in query_result)
        except Exception as e:
            print(f"An error occurred while fetching vectors: {e}")
            break

    if not vectors:
        raise ValueError("No vectors fetched. Check your query or collection configuration.")

    return np.array(vectors)

def pairwise_distance_matrix_gpu(vectors):
    """Compute the pairwise distance matrix on GPU using CuPy."""
    vectors_gpu = cp.asarray(vectors)  # Transfer data to GPU
    sq_dist_matrix = cp.sum(vectors_gpu ** 2, axis=1).reshape(-1, 1) \
                     + cp.sum(vectors_gpu ** 2, axis=1).reshape(1, -1) \
                     - 2 * cp.dot(vectors_gpu, vectors_gpu.T)
    distance_matrix = cp.sqrt(cp.maximum(sq_dist_matrix, 0))
    return cp.asnumpy(distance_matrix)  # Transfer result back to CPU if needed

def compute_distance_batches(vectors, batch_size=4096):
    """Compute the pairwise distance matrix in batches for large datasets."""
    n = vectors.shape[0]
    distances = np.zeros((n, n))  # Initialize full distance matrix

    print("Computing pairwise distances in batches with progress bar:")
    for i in tqdm(range(0, n, batch_size), desc="Outer Batches"):
        for j in range(0, n, batch_size):
            batch_i = vectors[i:i+batch_size]
            batch_j = vectors[j:j+batch_size]
            batch_distances = pairwise_distance_matrix_gpu(batch_i @ batch_j.T)
            distances[i:i+batch_size, j:j+batch_size] = batch_distances

    return distances

def compute_neighbor_values(vectors, k=5, batch_size=4096):
    """Compute the average distance to k-nearest neighbors for each vector."""
    n = vectors.shape[0]
    distances = compute_distance_batches(vectors, batch_size)

    neighbor_values = np.zeros(n)
    print("Calculating neighbor values with progress bar:")
    for i in tqdm(range(n), desc="Neighbors Progress"):
        nearest_indices = np.argsort(distances[i])[1:k+1]  # Get k nearest neighbors (exclude self)
        nearest_distances = distances[i, nearest_indices]
        neighbor_values[i] = np.mean(nearest_distances)

    return neighbor_values

def plot_3d_pointcloud_with_heatmap(vectors, neighbor_values):
    """Plot a 3D point cloud with a heatmap based on neighbor values."""
    if vectors.shape[1] > 3:
        print("Warning: High-dimensional data detected. Using only the first 3 dimensions.")
        vectors = vectors[:, :3]

    norm = plt.Normalize(neighbor_values.min(), neighbor_values.max())
    cmap = plt.cm.viridis  # Colormap

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        vectors[:, 0], vectors[:, 1], vectors[:, 2],
        s=5, c=neighbor_values, cmap=cmap, alpha=0.7
    )

    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    ax.set_title("3D Point Cloud with Heatmap")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    plt.show()

def main():
    """Main function to fetch vectors, compute neighbor values, and plot."""
    try:
        # Connect to Milvus
        collection = connect_to_milvus(MILVUS_CONFIG)
        print(f"Collection '{MILVUS_CONFIG['collection_name']}' contains {collection.num_entities} entities.")

        if collection.num_entities == 0:
            print("The collection is empty. Exiting.")
            return

        # Fetch vectors from Milvus
        vectors = fetch_vectors(collection, limit=16384)
        print(f"Fetched {vectors.shape[0]} vectors with {vectors.shape[1]} dimensions.")
        if vectors.size == 0:
            print("No vectors fetched. Exiting.")
            return

        # Compute average neighbor values
        print("Computing neighbor values for each vector.")
        neighbor_values = compute_neighbor_values(vectors, k=5)

        # Plot 3D point cloud with heatmap
        print("Plotting 3D point cloud with heatmap.")
        plot_3d_pointcloud_with_heatmap(vectors, neighbor_values)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
