"""Rank symptom-model probabilities against the training disease labels."""

from typing import Sequence


def rank_disease_probabilities(
    probabilities: Sequence[float],
    class_labels: Sequence[int],
    diseases: Sequence[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Return the highest-scoring model classes mapped to disease names."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if len(probabilities) != len(class_labels):
        raise ValueError("Probability and class-label counts must match.")

    ranked_indices = sorted(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
        reverse=True,
    )[:top_k]
    ranked = []
    for index in ranked_indices:
        disease_index = int(class_labels[index])
        if disease_index < 0 or disease_index >= len(diseases):
            raise ValueError(f"Model class label is outside the disease list: {disease_index}")
        ranked.append((str(diseases[disease_index]), float(probabilities[index])))
    return ranked
