import matplotlib.pyplot as plt
import numpy as np

def visualize_embeddings(embeddings, labels, n_components=2, title="Embeddings Visualization"):
    """
    Visualize high-dimensional embeddings using PCA from scratch.
    """
    # 1. Center the data
    mean = np.mean(embeddings, axis=0)
    centered_data = embeddings - mean
    
    # 2. Compute SVD
    U, S, Vh = np.linalg.svd(centered_data, full_matrices=False)
    
    # 3. Project data to n_components
    projected_data = np.dot(centered_data, Vh[:n_components].T)
    
    # 4. Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
    
    for i, label in enumerate(labels):
        plt.annotate(label, (projected_data[i, 0], projected_data[i, 1]))
    
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

def plot_training_curves(losses, accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title("Accuracy Curve")
    plt.show()
