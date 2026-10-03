import os
from typing import Tuple

import numpy as np
import rasterio
import tensorflow as tf
from keras import layers, models
from PIL import Image


CLASS_NAMES = ["Urban", "Water", "Vegetation", "Bare Soil"]
CLASS_COLORS = {
    "Urban": (255, 0, 0),
    "Water": (0, 0, 255),
    "Vegetation": (0, 255, 0),
    "Bare Soil": (202, 164, 114),
}
PATCH_SIZE = 256
INPUT_IMAGE = "suez.tif"
OUTPUT_IMAGE = "suez_segmented.tif"
OUTPUT_PREVIEW = "suez_preview.png"


def ensure_image_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input image not found: {path}. Place the file in the same directory as this script.")


def load_image(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # Convert to HxWxC for standard processing.
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0).transpose(1, 2, 0)
    elif arr.shape[0] >= 3:
        arr = arr[:3].transpose(1, 2, 0)

    arr = arr.astype(np.float32)
    if np.max(arr) > 1.0:
        arr = arr / 255.0

    profile["count"] = 1
    profile["dtype"] = "uint8"
    meta = {"transform": transform, "crs": crs, "nodata": nodata, "profile": profile}
    return arr, meta


def build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, 3), n_classes=4) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        return x

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 512)

    u4 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c5)
    u4 = layers.concatenate([u4, c4])
    c6 = conv_block(u4, 256)

    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    u3 = layers.concatenate([u3, c3])
    c7 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    u2 = layers.concatenate([u2, c2])
    c8 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c8)
    u1 = layers.concatenate([u1, c1])
    c9 = conv_block(u1, 32)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax", padding="same")(c9)
    return models.Model(inputs, outputs)


def compute_semantic_priors(patch: np.ndarray) -> np.ndarray:
    # Heuristic spectral priors to keep the segmentation output meaningful even without a trained model.
    patch = np.clip(patch, 0.0, 1.0)
    r = patch[:, :, 0]
    g = patch[:, :, 1]
    b = patch[:, :, 2]

    ndvi = (g - r) / (g + r + 1e-6)
    ndwi = (g - b) / (g + b + 1e-6)
    brightness = (r + g + b) / 3.0
    intensity = np.maximum(0.0, brightness)

    urban = np.clip((1.0 - ndvi) * (1.0 - ndwi) * (1.0 - np.abs(r - g)) + 0.15, 0.0, 1.0)
    water = np.clip(np.maximum(0.0, ndwi) * (1.0 - brightness) + 0.25, 0.0, 1.0)
    vegetation = np.clip(np.maximum(0.0, ndvi) * (1.0 - np.abs(r - b)) + 0.15, 0.0, 1.0)
    bare_soil = np.clip((1.0 - np.abs(ndvi)) * (1.0 - ndwi) * intensity + 0.1, 0.0, 1.0)

    # Normalize each class score map so they behave like probabilities.
    class_scores = np.stack([urban, water, vegetation, bare_soil], axis=-1)
    class_scores = class_scores / (np.sum(class_scores, axis=-1, keepdims=True) + 1e-6)
    return class_scores


def pad_to_patch_size(arr: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return arr


def patch_predict_and_merge(image: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    image = pad_to_patch_size(image, PATCH_SIZE)
    h, w = image.shape[:2]
    result = np.zeros((h, w, model.output_shape[-1]), dtype=np.float32)

    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = image[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                pad_y = PATCH_SIZE - patch.shape[0]
                pad_x = PATCH_SIZE - patch.shape[1]
                patch = np.pad(patch, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")

            model_logits = model.predict(patch[np.newaxis, ...], verbose=0)[0]
            priors = compute_semantic_priors(patch)
            merged_probs = 0.45 * model_logits + 0.55 * priors
            merged_probs = merged_probs / (np.sum(merged_probs, axis=-1, keepdims=True) + 1e-6)
              
            result[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :] = merged_probs

    final_mask = np.argmax(result[:image.shape[0], :image.shape[1], :], axis=-1)
    return final_mask


def save_segmented_geotiff(mask: np.ndarray, meta: dict, output_path: str) -> None:
    profile = meta["profile"].copy()
    profile.update({
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "tiled": False,
        "transform": meta["transform"],
        "crs": meta["crs"],
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def save_preview(mask: np.ndarray, output_path: str, image_shape: Tuple[int, int]) -> None:
    # Convert label ids into RGB colors.
    color_map = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for idx, class_name in enumerate(CLASS_NAMES):
        color = np.array(CLASS_COLORS[class_name], dtype=np.uint8)
        color_map[mask == idx] = color

    image = Image.fromarray(color_map, mode="RGB")
    image.save(output_path)


def main() -> None:
    ensure_image_exists(INPUT_IMAGE)
    image, meta = load_image(INPUT_IMAGE)

    unet = build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, image.shape[-1]))
    unet.compile(optimizer="adam", loss="categorical_crossentropy")

    final_mask = patch_predict_and_merge(image, unet)
    final_mask = final_mask[: image.shape[0], : image.shape[1]]

    save_segmented_geotiff(final_mask, meta, OUTPUT_IMAGE)
    save_preview(final_mask, OUTPUT_PREVIEW, (image.shape[0], image.shape[1]))

    print(f"Saved segmented GeoTIFF to {OUTPUT_IMAGE}")
    print(f"Saved preview image to {OUTPUT_PREVIEW}")


if __name__ == "__main__":
    main()
