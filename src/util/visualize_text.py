import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import cupy as cp


# Text Embedding Processor
class TextEmbeddingProcessor:
    def __init__(self, api_url):
        self.api_url = api_url

    def get_embedding(self, text):
        """Fetch embedding for a given text."""
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={"model": "snowflake-arctic-embed2:latest", "input": text},
            )
            response.raise_for_status()
            data = response.json()
            if "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]
            elif "embeddings" in data and isinstance(data["embeddings"], list) and len(data["embeddings"]) > 0:
                return data["embeddings"][0]
            else:
                print(f"Unexpected API response format for text: {text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed for text '{text}': {e}")
            return None

    def process_file(self, file_path):
        """Read input file and fetch embeddings for each line."""
        embeddings = []
        texts = []
        with open(file_path, 'r') as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                embedding = self.get_embedding(text)
                if embedding:
                    embeddings.append(embedding)
                    texts.append(text)
        return texts, embeddings


# GPU-accelerated Pairwise Distance Calculation
def pairwise_distance_matrix_gpu(vectors):
    """Compute the pairwise distance matrix on GPU using CuPy."""
    vectors_gpu = cp.asarray(vectors)  # Transfer data to GPU
    sq_dist_matrix = cp.sum(vectors_gpu ** 2, axis=1).reshape(-1, 1) \
                     + cp.sum(vectors_gpu ** 2, axis=1).reshape(1, -1) \
                     - 2 * cp.dot(vectors_gpu, vectors_gpu.T)
    distance_matrix = cp.sqrt(cp.maximum(sq_dist_matrix, 0))
    return cp.asnumpy(distance_matrix)  # Transfer result back to CPU if needed


# Compute Pairwise Distances in Batches
def compute_distance_batches(vectors, batch_size=4096):
    """Compute the pairwise distance matrix in batches for large datasets."""
    n = vectors.shape[0]
    distances = np.zeros((n, n))  # Initialize full distance matrix

    print("Computing pairwise distances in batches with progress bar:")
    for i in tqdm(range(0, n, batch_size), desc="Outer Batches"):
        for j in range(0, n, batch_size):
            batch_i = vectors[i:i + batch_size]
            batch_j = vectors[j:j + batch_size]
            batch_distances = pairwise_distance_matrix_gpu(batch_i @ batch_j.T)
            distances[i:i + batch_size, j:j + batch_size] = batch_distances

    return distances


# Compute Average Neighbor Values
def compute_neighbor_values(vectors, k=5, batch_size=4096):
    """Compute the average distance to k-nearest neighbors for each vector."""
    n = vectors.shape[0]
    distances = compute_distance_batches(vectors, batch_size)

    neighbor_values = np.zeros(n)
    print("Calculating neighbor values with progress bar:")
    for i in tqdm(range(n), desc="Neighbors Progress"):
        nearest_indices = np.argsort(distances[i])[1:k + 1]  # Get k nearest neighbors (exclude self)
        nearest_distances = distances[i, nearest_indices]
        neighbor_values[i] = np.mean(nearest_distances)

    return neighbor_values


# 3D Point Cloud Visualization
def plot_3d_pointcloud_with_labels(vectors, neighbor_values, labels=None):
    """
    Plot a 3D point cloud with heatmap, ensuring all points are displayed,
    and attaching labels only to the specified points.
    """
    if vectors.shape[1] > 3:
        print("Warning: High-dimensional data detected. Using only the first 3 dimensions.")
        vectors = vectors[:, :3]

    # Normalize the neighbor values for the colormap
    norm = plt.Normalize(neighbor_values.min(), neighbor_values.max())
    cmap = plt.cm.viridis  # Colormap

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points with heatmap
    scatter = ax.scatter(
        vectors[:, 0], vectors[:, 1], vectors[:, 2],
        s=5, c=neighbor_values, cmap=cmap, alpha=0.7
    )

    # Add labels to specific points if labels are provided
    if labels:
        for i, label in enumerate(labels):
            if label:  # Only add a label if it exists
                ax.text(vectors[i, 0], vectors[i, 1], vectors[i, 2], label, size=6, alpha=0.8)

    # Add a color bar for the heatmap
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)

    # Set titles and axis labels
    ax.set_title("3D Point Cloud with Heatmap and Labels")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")

    # Show the plot
    plt.show()


# Main Pipeline
def main(file_path, api_url):
    """Main function to process file, compute neighbor values, and plot."""
    processor = TextEmbeddingProcessor(api_url)

    print(f"Processing file: {file_path}")
    texts, embeddings = processor.process_file(file_path)
    if not embeddings:
        print("No embeddings generated. Exiting.")
        return

    embeddings = np.array(embeddings)
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}.")

    # Compute average neighbor values
    print("Computing neighbor values for each embedding.")
    neighbor_values = compute_neighbor_values(embeddings, k=5)

    # Plot 3D point cloud with heatmap
    print("Plotting 3D point cloud with heatmap.")
    plot_3d_pointcloud_with_labels(embeddings, neighbor_values, texts)


if __name__ == "__main__":
    INPUT_FILE = "input.txt"  # Input file containing lines of text
    API_URL = "http://127.0.0.1:11434/api/embed"  # Replace with actual embedding API URL
    main(INPUT_FILE, API_URL)
