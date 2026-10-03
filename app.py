import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Configuration
INPUT_IMAGE = "suez.jpg"
OUTPUT_IMAGE = "suez_segmented.png"
TARGET_SIZE = (1024, 1024)
N_CLUSTERS = 4

# Class names for the 4 clusters
CLASS_NAMES = ["Urban", "Water", "Vegetation", "Bare Soil"]

# Color mapping for each cluster (RGB)
CLASS_COLORS = {
    0: (128, 128, 128),    # Urban - Gray
    1: (0, 0, 255),        # Water - Blue
    2: (0, 255, 0),        # Vegetation - Green
    3: (255, 255, 0),      # Bare Soil - Yellow
}


def ensure_image_exists(path: str) -> None:
    """Check if input image exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input image not found: {path}. Place the file in the same directory as this script."
        )


def load_and_preprocess_image(path: str, target_size: tuple) -> np.ndarray:
    """Load image and resize to target size."""
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def perform_kmeans_segmentation(image: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Apply KMeans clustering to segment the image.
    
    Args:
        image: Input image array (H, W, 3) with values in [0, 1]
        n_clusters: Number of clusters
    
    Returns:
        Segmentation mask (H, W) with cluster labels
    """
    h, w, c = image.shape
    
    # Reshape image to (H*W, C) for clustering
    pixels = image.reshape(-1, c)
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape back to image dimensions
    mask = labels.reshape(h, w)
    
    return mask


def create_colored_segmentation(mask: np.ndarray, colors: dict, target_size: tuple) -> Image.Image:
    """
    Create a colored segmentation image from mask labels.
    
    Args:
        mask: Segmentation mask with cluster labels
        colors: Dictionary mapping cluster index to RGB color
        target_size: Target image size
    
    Returns:
        PIL Image with colored segmentation
    """
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for cluster_id in range(len(colors)):
        colored[mask == cluster_id] = colors[cluster_id]
    
    return Image.fromarray(colored, mode="RGB")


def save_segmentation_results(
    mask: np.ndarray,
    colors: dict,
    output_path: str,
    target_size: tuple
) -> None:
    """Save colored segmentation image."""
    colored_img = create_colored_segmentation(mask, colors, target_size)
    colored_img.save(output_path)
    print(f"Saved colored segmentation to {output_path}")


def print_segmentation_info(mask: np.ndarray) -> None:
    """Print information about the segmentation results."""
    print("\nSegmentation Results:")
    print("-" * 40)
    for cluster_id in range(len(CLASS_NAMES)):
        pixels = np.sum(mask == cluster_id)
        percentage = (pixels / mask.size) * 100
        print(f"{CLASS_NAMES[cluster_id]}: {percentage:.2f}% ({pixels} pixels)")


def main() -> None:
    """Main segmentation pipeline."""
    print("Loading image...")
    ensure_image_exists(INPUT_IMAGE)
    
    # Load and preprocess image
    image = load_and_preprocess_image(INPUT_IMAGE, TARGET_SIZE)
    print(f"Image loaded and resized to {TARGET_SIZE}")
    
    # Perform KMeans clustering
    print("Performing KMeans clustering...")
    mask = perform_kmeans_segmentation(image, N_CLUSTERS)
    
    # Save results
    print("Creating colored segmentation...")
    save_segmentation_results(mask, CLASS_COLORS, OUTPUT_IMAGE, TARGET_SIZE)
    
    # Print statistics
    print_segmentation_info(mask)
    
    print("\nSegmentation complete!")


if __name__ == "__main__":
    main()