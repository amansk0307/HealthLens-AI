"""Inference helpers for the Alzheimer's MRI classification model."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "alzheimer_model.h5"
IMAGE_SIZE = (244, 244)
CLASS_NAMES = (
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very MildDemented",
)


def load_alzheimer_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Alzheimer model file not found: {model_path}")

    base_model = tf.keras.applications.DenseNet201(
        weights=None,
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        pooling=None,
    )
    features = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    features = tf.keras.layers.Dense(1024, activation="relu")(features)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(features)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.load_weights(model_path)
    return model


def predict_alzheimer_image(
    image: Image.Image, model: tf.keras.Model
) -> tuple[str, float]:
    pixels = np.asarray(
        image.convert("RGB").resize(IMAGE_SIZE, resample=Image.Resampling.NEAREST),
        dtype=np.float32,
    )
    pixels = tf.keras.applications.mobilenet_v2.preprocess_input(pixels)
    probabilities = model.predict(np.expand_dims(pixels, axis=0), verbose=0)[0]
    if probabilities.shape != (len(CLASS_NAMES),):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class scores, got {probabilities.shape}."
        )

    predicted_index = int(np.argmax(probabilities))
    return CLASS_NAMES[predicted_index], float(probabilities[predicted_index])
