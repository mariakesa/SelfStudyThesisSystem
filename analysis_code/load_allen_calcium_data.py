"""Initial data node: raw Allen natural-scenes calcium imaging -> canonical datasets.

For every ophys session showing the `natural_scenes` stimulus, build a
neurons x (images x trials) binary event-response matrix, then aggregate
across sessions into two canonical composite datasets:

- a trial-averaged probability matrix (neurons x n_images), via
  `assemble_probability_composite`
- a raw, un-averaged binary trial matrix (neurons x (n_images * n_trials)),
  via `assemble_binary_trial_composite`

Every function here takes its inputs explicitly and has no side effects
beyond `save_composite_dataset`. Nothing runs at import time; a caller
(a notebook, or a later run script) is responsible for chaining
`build_natural_scenes_binary_dataset` -> `assemble_*_composite` ->
`save_composite_dataset` against a real Allen manifest.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.allen_client import get_brain_observatory_cache, resolve_manifest_path

LABEL_NAMES = {
    -1: "unlabeled",
    0: "animals",
    1: "landscape",
    2: "plant",
    3: "man-made object",
}


# ---------------------------------------------------------------------------
# Session discovery (I/O)
# ---------------------------------------------------------------------------

def list_natural_scenes_sessions(boc: Any, stimulus: str = "natural_scenes") -> list[dict]:
    """Return Allen experiment records showing `stimulus`, sorted by experiment id."""
    experiments = boc.get_ophys_experiments(stimuli=[stimulus])
    return sorted(experiments, key=lambda experiment: int(experiment["id"]))


# ---------------------------------------------------------------------------
# Pure computation (no I/O)
# ---------------------------------------------------------------------------

def _stimulus_table_columns(stim_table: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull `frame`, `start`, `end` columns from a stim_table as int64 arrays."""

    def _column(name: str) -> np.ndarray:
        column = stim_table[name]
        if hasattr(column, "to_numpy"):
            column = column.to_numpy()
        return np.asarray(column, dtype=np.int64)

    return _column("frame"), _column("start"), _column("end")


def compute_binary_trial_matrix(
    events: np.ndarray,
    stim_table: Any,
    *,
    n_images: int = 118,
    n_trials_per_image: int = 50,
    threshold: float = 0.0,
) -> np.ndarray:
    """Build one session's neurons x (n_images * n_trials_per_image) binary matrix.

    For each stimulus presentation, take the L0 event trace over the
    presentation window `events[:, start:end]`, threshold its per-cell max,
    and record a binary (n_cells,) indicator. Columns are image-major:
    image 0 trial 0..n_trials_per_image-1, image 1 trial 0..n_trials_per_image-1, ...

    Assumes every image has at least `n_trials_per_image` presentations in
    `stim_table` (true for all sessions on this dataset); raises ValueError
    otherwise rather than silently padding or dropping the session.
    """
    frame_col, start_col, end_col = _stimulus_table_columns(stim_table)
    n_cells = events.shape[0]

    frame_trials: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame_idx, start_t, end_t in zip(frame_col, start_col, end_col):
        if frame_idx == -1:
            continue
        if end_t <= start_t:
            trial_vector = np.zeros(n_cells, dtype=np.float64)
        else:
            window = events[:, start_t:end_t]
            trial_vector = (window.max(axis=1) > threshold).astype(np.float64)
        frame_trials[int(frame_idx)].append(trial_vector)

    matrix = np.empty((n_cells, n_images * n_trials_per_image), dtype=np.float64)
    for image_idx in range(n_images):
        trials = frame_trials.get(image_idx, [])
        if len(trials) < n_trials_per_image:
            raise ValueError(
                f"image {image_idx} has only {len(trials)} trials, "
                f"expected at least {n_trials_per_image}"
            )
        for trial_idx in range(n_trials_per_image):
            column = image_idx * n_trials_per_image + trial_idx
            matrix[:, column] = trials[trial_idx]

    return matrix


def trial_matrix_to_probability_matrix(
    matrix: np.ndarray, *, n_images: int = 118, n_trials_per_image: int = 50
) -> np.ndarray:
    """Average a neurons x (n_images * n_trials) binary matrix into neurons x n_images.

    Each output value is P(neuron active | image), the mean binary response
    across that image's trials.
    """
    n_cells, n_columns = matrix.shape
    expected_columns = n_images * n_trials_per_image
    if n_columns != expected_columns:
        raise ValueError(f"expected {expected_columns} columns, got {n_columns}")

    reshaped = matrix.reshape(n_cells, n_images, n_trials_per_image)
    return reshaped.mean(axis=2)


# ---------------------------------------------------------------------------
# Metadata (I/O)
# ---------------------------------------------------------------------------

def get_experiment_metadata_table(boc: Any) -> pd.DataFrame:
    """Return Allen experiment metadata indexed by ophys experiment id."""
    experiments = pd.DataFrame(boc.get_ophys_experiments())
    if "id" not in experiments.columns:
        raise KeyError("Allen experiment table does not contain column 'id'.")
    return experiments.set_index("id", drop=False)


def _get_area_for_session(session_id: int, experiment_table: pd.DataFrame, dataset: Any) -> str:
    """Look up the targeted brain area for a session, falling back to NWB metadata."""
    if session_id in experiment_table.index:
        row = experiment_table.loc[session_id]
        for column in ("targeted_structure", "structure", "imaging_area"):
            if column in row and pd.notna(row[column]):
                return str(row[column])

    try:
        metadata = dataset.get_metadata()
        for key in ("targeted_structure", "structure", "imaging_area"):
            if key in metadata and metadata[key] is not None:
                return str(metadata[key])
    except Exception:
        pass

    return "unknown"


def get_session_metadata(session_id: int, experiment_table: pd.DataFrame, dataset: Any) -> dict:
    """Collect session-level Allen metadata (brain area, cre line, NWB fields, ...)."""
    meta: dict[str, Any] = {
        "ophys_experiment_id": int(session_id),
        "brain_area": _get_area_for_session(session_id, experiment_table, dataset),
    }

    if session_id in experiment_table.index:
        row = experiment_table.loc[session_id]
        for column in (
            "experiment_container_id",
            "session_type",
            "targeted_structure",
            "imaging_depth",
            "cre_line",
            "reporter_line",
            "specimen_name",
            "donor_name",
        ):
            if column in experiment_table.columns:
                value = row[column]
                if pd.isna(value):
                    value = None
                if isinstance(value, np.generic):
                    value = value.item()
                meta[column] = value

    try:
        nwb_metadata = dataset.get_metadata()
        for key, value in nwb_metadata.items():
            if key not in meta:
                if isinstance(value, np.generic):
                    value = value.item()
                meta[key] = value
    except Exception:
        pass

    return meta


def load_four_class_image_labels(label_path: Path, *, n_images: int = 118) -> np.ndarray:
    """Load per-image four-class labels (see LABEL_NAMES) from a saved .npy file.

    Handles either a plain (n_images,) label array or a saved dict with key
    "labels".
    """
    obj = np.load(label_path, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()

    if isinstance(obj, dict):
        if "labels" not in obj:
            raise KeyError(
                f"{label_path} is a dict but does not contain key 'labels'. "
                f"Available keys: {list(obj.keys())}"
            )
        labels = np.asarray(obj["labels"])
    else:
        labels = np.asarray(obj)

    if labels.shape[0] != n_images:
        raise ValueError(f"Expected {n_images} image labels, got shape {labels.shape}")

    labels = labels.astype(int)

    valid_values = set(LABEL_NAMES.keys())
    observed_values = set(np.unique(labels).tolist())
    unknown_values = observed_values - valid_values
    if unknown_values:
        raise ValueError(
            f"Image labels contain unknown label values: {unknown_values}. "
            f"Expected only {valid_values}"
        )

    return labels


# ---------------------------------------------------------------------------
# Per-session loading (I/O)
# ---------------------------------------------------------------------------

def load_session_binary_responses(
    session_id: int,
    boc: Any,
    experiment_table: pd.DataFrame,
    *,
    stimulus: str = "natural_scenes",
    n_images: int = 118,
    n_trials_per_image: int = 50,
    threshold: float = 0.0,
) -> dict:
    """Load one session's raw data and compute its binary trial-response matrix."""
    dataset = boc.get_ophys_experiment_data(session_id)
    events = np.asarray(boc.get_ophys_experiment_events(session_id))
    stim_table = dataset.get_stimulus_table(stimulus)
    cell_specimen_ids = np.asarray(dataset.get_cell_specimen_ids())

    matrix = compute_binary_trial_matrix(
        events,
        stim_table,
        n_images=n_images,
        n_trials_per_image=n_trials_per_image,
        threshold=threshold,
    )

    metadata = get_session_metadata(session_id, experiment_table, dataset)

    return {
        "session_id": int(session_id),
        "matrix": matrix,
        "cell_specimen_ids": cell_specimen_ids,
        "n_cells": int(matrix.shape[0]),
        "brain_area": metadata["brain_area"],
        "metadata": metadata,
    }


def build_natural_scenes_binary_dataset(
    *,
    manifest_path: Path | str | None = None,
    stimulus: str = "natural_scenes",
    n_images: int = 118,
    n_trials_per_image: int = 50,
    threshold: float = 0.0,
    session_limit: int | None = None,
) -> dict:
    """Load every natural-scenes session's binary trial-response matrix.

    Returns {"stimulus", "n_images", "n_trials_per_image", "threshold", "sessions"}
    where "sessions" is a list of per-session records from
    `load_session_binary_responses`.
    """
    resolved_manifest_path = resolve_manifest_path(manifest_path)
    boc = get_brain_observatory_cache(resolved_manifest_path)

    experiments = list_natural_scenes_sessions(boc, stimulus=stimulus)
    if session_limit is not None:
        experiments = experiments[:session_limit]

    experiment_table = get_experiment_metadata_table(boc)

    sessions = [
        load_session_binary_responses(
            int(experiment["id"]),
            boc,
            experiment_table,
            stimulus=stimulus,
            n_images=n_images,
            n_trials_per_image=n_trials_per_image,
            threshold=threshold,
        )
        for experiment in experiments
    ]

    return {
        "stimulus": stimulus,
        "n_images": n_images,
        "n_trials_per_image": n_trials_per_image,
        "threshold": threshold,
        "sessions": sessions,
    }


# ---------------------------------------------------------------------------
# Pure aggregation (no I/O)
# ---------------------------------------------------------------------------

def build_neuron_metadata(sessions: list[dict]) -> dict:
    """Concatenate per-neuron metadata row-aligned across sessions."""
    cell_specimen_ids: list[int] = []
    brain_areas: list[str] = []
    session_ids: list[int] = []
    local_indices: list[int] = []

    for session in sessions:
        n_cells = session["n_cells"]
        cell_specimen_ids.extend(np.asarray(session["cell_specimen_ids"], dtype=int).tolist())
        brain_areas.extend([session["brain_area"]] * n_cells)
        session_ids.extend([session["session_id"]] * n_cells)
        local_indices.extend(range(n_cells))

    return {
        "cell_specimen_id": np.asarray(cell_specimen_ids, dtype=int),
        "brain_area": np.asarray(brain_areas, dtype=object),
        "ophys_experiment_id": np.asarray(session_ids, dtype=int),
        "local_neuron_index": np.asarray(local_indices, dtype=int),
        "description": (
            "Neuron-level metadata aligned row-by-row with X. For row i in X, "
            "use neuron_metadata['cell_specimen_id'][i], "
            "neuron_metadata['brain_area'][i], etc."
        ),
    }


def build_stimulus_metadata_probability(labels: np.ndarray, label_names: dict = LABEL_NAMES) -> dict:
    """Build column-aligned stimulus metadata for the neurons x n_images composite."""
    n_images = labels.shape[0]
    return {
        "image_index": np.arange(n_images, dtype=int),
        "label": labels,
        "label_name": np.asarray([label_names[int(x)] for x in labels], dtype=object),
        "label_names": label_names,
        "description": (
            "Stimulus metadata aligned column-by-column with X. For column j "
            "in X, use stimulus_metadata['label'][j] and "
            "stimulus_metadata['label_name'][j]."
        ),
    }


def build_stimulus_metadata_trial_level(
    labels: np.ndarray,
    label_names: dict = LABEL_NAMES,
    *,
    n_images: int = 118,
    n_trials_per_image: int = 50,
) -> dict:
    """Build column-aligned stimulus metadata for the neurons x (n_images * n_trials) composite."""
    image_index_by_column = np.repeat(np.arange(n_images, dtype=int), n_trials_per_image)
    trial_index_by_column = np.tile(np.arange(n_trials_per_image, dtype=int), n_images)
    label_by_column = labels[image_index_by_column]

    return {
        "column_index": np.arange(n_images * n_trials_per_image, dtype=int),
        "image_index": image_index_by_column,
        "trial_index": trial_index_by_column,
        "label": label_by_column,
        "label_name": np.asarray([label_names[int(x)] for x in label_by_column], dtype=object),
        "image_label": labels,
        "image_label_name": np.asarray([label_names[int(x)] for x in labels], dtype=object),
        "label_names": label_names,
        "description": (
            "Stimulus metadata aligned column-by-column with X. X has "
            "total_neurons x (n_images * n_trials_per_image) columns. For "
            "column k in X, use stimulus_metadata['image_index'][k], "
            "stimulus_metadata['trial_index'][k], stimulus_metadata['label'][k], "
            "and stimulus_metadata['label_name'][k]. Image-level labels are "
            "also stored as stimulus_metadata['image_label']."
        ),
    }


def assemble_probability_composite(
    sessions: list[dict],
    labels: np.ndarray,
    *,
    label_names: dict = LABEL_NAMES,
    n_images: int = 118,
    n_trials_per_image: int = 50,
) -> dict:
    """Aggregate per-session binary matrices into one trial-averaged composite dataset."""
    probability_matrices = [
        trial_matrix_to_probability_matrix(
            session["matrix"], n_images=n_images, n_trials_per_image=n_trials_per_image
        )
        for session in sessions
    ]
    X = np.vstack(probability_matrices)

    return {
        "X": X,
        "neuron_metadata": build_neuron_metadata(sessions),
        "stimulus_metadata": build_stimulus_metadata_probability(labels, label_names),
        "label_names": label_names,
        "session_metadata": {session["session_id"]: session["metadata"] for session in sessions},
        "session_ids_used": [session["session_id"] for session in sessions],
        "constants": {"n_images": n_images, "n_trials_per_image": n_trials_per_image},
        "description": (
            f"Composite Allen natural-scenes dataset. X is total_neurons x {n_images}. "
            "Rows are neurons concatenated across ophys sessions. Columns are "
            "natural scene images. X[i, j] is the average binary event response "
            "across trials, interpreted as P(neuron active | image). Neuron "
            "metadata is row-aligned with X. Stimulus metadata is column-aligned "
            "with X."
        ),
    }


def assemble_binary_trial_composite(
    sessions: list[dict],
    labels: np.ndarray,
    *,
    label_names: dict = LABEL_NAMES,
    n_images: int = 118,
    n_trials_per_image: int = 50,
) -> dict:
    """Aggregate per-session binary matrices into one raw (un-averaged) composite dataset."""
    X = np.vstack([session["matrix"] for session in sessions])

    return {
        "X": X,
        "neuron_metadata": build_neuron_metadata(sessions),
        "stimulus_metadata": build_stimulus_metadata_trial_level(
            labels, label_names, n_images=n_images, n_trials_per_image=n_trials_per_image
        ),
        "label_names": label_names,
        "session_metadata": {session["session_id"]: session["metadata"] for session in sessions},
        "session_ids_used": [session["session_id"] for session in sessions],
        "constants": {
            "n_images": n_images,
            "n_trials_per_image": n_trials_per_image,
            "column_order": (
                "image-major: image_index = column // n_trials_per_image; "
                "trial_index = column % n_trials_per_image"
            ),
        },
        "description": (
            "Composite Allen natural-scenes binary trial-response dataset. X "
            f"is total_neurons x {n_images * n_trials_per_image}. Rows are "
            "neurons concatenated across ophys sessions. Columns are individual "
            "natural-scene presentations in image-major order. X[i, k] is the "
            "original binary event response for neuron i on presentation "
            "column k; trials are not averaged into probabilities. Neuron "
            "metadata is row-aligned with X. Stimulus metadata is "
            "column-aligned with X."
        ),
    }


# ---------------------------------------------------------------------------
# Persistence (I/O)
# ---------------------------------------------------------------------------

def save_composite_dataset(composite: dict, output_path: Path | str) -> None:
    """Save a composite dataset dict to `output_path` as a pickled .npy file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, composite, allow_pickle=True)
