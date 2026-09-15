#!/usr/bin/env python3
"""Interactive GUI for labeling Allen natural-scene stimulus images into four classes.

Ported from /home/maria/Science/thesis/experiments/010--TransformerErrorAnalysis/
image_labeler_4_classes_gui.py. Produces the same label file format consumed by
`load_allen_calcium_data.load_four_class_image_labels`: a pickled .npy dict with
keys "labels" (int64 array, -1=unlabeled) and "image_paths", plus a JSON sidecar
for human inspection.

This is a standalone interactive tool, not a pure analysis function -- run it
directly:

    python analysis_code/label_stimulus_images_gui.py \\
        --image-dir /path/to/images --output /path/to/four_class_image_labels.npy

Keys while the window is focused: 0=animals, 1=landscape, 2=plant,
3=man-made object, arrows/space=navigate, c=clear label, s=save, q=quit.

Reuses LABEL_NAMES from `load_allen_calcium_data` so the label convention has
one source of truth (the original Science repo had two GUI copies with
inconsistent, even duplicated, LABEL_NAMES definitions -- not repeated here).
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

from analysis_code.load_allen_calcium_data import LABEL_NAMES

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Pure logic (no I/O, no Tk) -- kept separate so it's testable without a display.
# ---------------------------------------------------------------------------

def compute_label_counts(labels: np.ndarray, label_names: dict = LABEL_NAMES) -> dict[str, int]:
    """Count how many images currently hold each label value."""
    return {
        label_names[value]: int(np.sum(labels == value))
        for value in sorted(label_names)
    }


def build_label_payload(
    image_dir: Path, image_paths: list[Path], labels: np.ndarray, label_names: dict = LABEL_NAMES
) -> dict:
    """Build the .npy-serializable label payload for one labeling session."""
    return {
        "image_dir": str(image_dir),
        "image_paths": np.array([str(p) for p in image_paths], dtype=object),
        "image_names": np.array([p.name for p in image_paths], dtype=object),
        "labels": labels.astype(np.int64),
        "label_names": label_names,
    }


def build_label_sidecar(
    image_dir: Path,
    output_path: Path,
    image_paths: list[Path],
    labels: np.ndarray,
    label_names: dict = LABEL_NAMES,
) -> dict:
    """Build the human-readable JSON sidecar for one labeling session."""
    return {
        "image_dir": str(image_dir),
        "output_path": str(output_path),
        "label_convention": {str(k): v for k, v in label_names.items()},
        "counts": compute_label_counts(labels, label_names),
        "items": [
            {
                "image": p.name,
                "path": str(p),
                "label": int(label),
                "label_name": label_names.get(int(label), f"unknown label {int(label)}"),
            }
            for p, label in zip(image_paths, labels)
        ],
    }


def save_labels_to_disk(payload: dict, sidecar: dict, output_path: Path) -> Path:
    """Write the .npy payload and its .json sidecar; returns the sidecar path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, payload, allow_pickle=True)

    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar_path


def find_images(image_dir: Path) -> list[Path]:
    """List labelable image files in `image_dir`, sorted by filename."""
    return sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name,
    )


def load_existing_labels(output_path: Path, image_paths: list[Path]) -> np.ndarray | None:
    """Load a prior label array from `output_path` if it matches `image_paths`.

    Returns None (rather than raising) if there is no existing file, its image
    list doesn't match the current directory, or it can't be read -- the
    caller decides whether/how to surface that to the user.
    """
    if not output_path.exists():
        return None

    saved = np.load(output_path, allow_pickle=True).item()
    saved_names = [Path(p).name for p in saved["image_paths"]]
    current_names = [p.name for p in image_paths]

    if saved_names != current_names:
        return None

    labels = np.asarray(saved["labels"], dtype=np.int64)
    if labels.shape != (len(image_paths),):
        return None

    return labels


# ---------------------------------------------------------------------------
# GUI (Tk) -- thin shell: rendering and widget state only, delegates to the
# pure/I/O functions above for everything else.
# ---------------------------------------------------------------------------

class ImageLabeler:
    def __init__(
        self,
        root: tk.Tk,
        image_dir: Path,
        output_path: Path,
        max_width: int = 1000,
        max_height: int = 750,
    ):
        self.root = root
        self.image_dir = image_dir.expanduser().resolve()
        self.output_path = output_path.expanduser().resolve()
        self.max_width = max_width
        self.max_height = max_height

        self.image_paths = find_images(self.image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        self.labels = np.full(len(self.image_paths), -1, dtype=np.int64)
        self.index = 0
        self.photo: ImageTk.PhotoImage | None = None

        self._load_existing_labels()
        self._build_ui()
        self._bind_keys()
        self._show_current_image()

    def _load_existing_labels(self) -> None:
        try:
            existing = load_existing_labels(self.output_path, self.image_paths)
        except Exception as exc:
            messagebox.showwarning(
                "Could not load existing labels",
                f"Could not read {self.output_path}.\n\nStarting fresh.\n\nError: {exc}",
            )
            return

        if existing is None:
            if self.output_path.exists():
                messagebox.showwarning(
                    "Label file not reused",
                    "Existing label file doesn't match the current image folder "
                    "(different images or count). Starting a fresh label array.",
                )
            return

        self.labels = existing
        first_unlabeled = np.where(self.labels == -1)[0]
        self.index = int(first_unlabeled[0]) if len(first_unlabeled) else 0

    def _build_ui(self) -> None:
        self.root.title("Image Labeler: animals / landscape / plant / man-made object")

        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(fill=tk.X, padx=10, pady=8)

        self.status_label = tk.Label(self.top_frame, text="", font=("Arial", 13))
        self.status_label.pack(side=tk.LEFT)

        self.progress_label = tk.Label(self.top_frame, text="", font=("Arial", 11))
        self.progress_label.pack(side=tk.RIGHT)

        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, padx=10, pady=8)

        tk.Button(self.button_frame, text="← Previous", command=self.previous_image, width=14).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(
            self.button_frame, text="Animals [0]", command=lambda: self.set_label(0), width=18
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.button_frame, text="Landscape [1]", command=lambda: self.set_label(1), width=18
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.button_frame, text="Plant [2]", command=lambda: self.set_label(2), width=18
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.button_frame, text="Man-made [3]", command=lambda: self.set_label(3), width=18
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(self.button_frame, text="Clear [C]", command=self.clear_label, width=14).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(self.button_frame, text="Next →", command=self.next_image, width=14).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(self.button_frame, text="Save [S]", command=self.save_labels, width=12).pack(
            side=tk.RIGHT, padx=4
        )

        self.help_label = tk.Label(
            self.root,
            text=(
                "Keys: 0=animals, 1=landscape, 2=plant, 3=man-made object, "
                "←/→=navigate, space=next, c=clear, s=save, q=quit"
            ),
            font=("Arial", 10),
        )
        self.help_label.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda event: self.previous_image())
        self.root.bind("<Right>", lambda event: self.next_image())
        self.root.bind("<space>", lambda event: self.next_image())

        for value in (0, 1, 2, 3):
            self.root.bind(str(value), lambda event, v=value: self.set_label(v))

        for key in ("c", "C"):
            self.root.bind(key, lambda event: self.clear_label())
        for key in ("s", "S"):
            self.root.bind(key, lambda event: self.save_labels())
        for key in ("q", "Q"):
            self.root.bind(key, lambda event: self.quit())

    def _show_current_image(self) -> None:
        path = self.image_paths[self.index]

        image = Image.open(path).convert("RGB")
        image.thumbnail((self.max_width, self.max_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

        label_value = int(self.labels[self.index])
        label_text = LABEL_NAMES.get(label_value, f"unknown label {label_value}")
        self.status_label.configure(
            text=f"{self.index + 1}/{len(self.image_paths)}  {path.name}  |  label: {label_text}"
        )

        n_labeled = int(np.sum(self.labels != -1))
        counts = compute_label_counts(self.labels)
        self.progress_label.configure(
            text=(
                f"Labeled: {n_labeled}/{len(self.labels)} | "
                f"animals={counts['animals']}, landscape={counts['landscape']}, "
                f"plant={counts['plant']}, man-made={counts['man-made object']}"
            )
        )

    def set_label(self, value: int) -> None:
        if value not in LABEL_NAMES or value == -1:
            raise ValueError(f"Invalid label value: {value}")

        self.labels[self.index] = value
        self.save_labels(show_popup=False)

        if self.index < len(self.image_paths) - 1:
            self.index += 1

        self._show_current_image()

    def clear_label(self) -> None:
        self.labels[self.index] = -1
        self.save_labels(show_popup=False)
        self._show_current_image()

    def previous_image(self) -> None:
        if self.index > 0:
            self.index -= 1
        self._show_current_image()

    def next_image(self) -> None:
        if self.index < len(self.image_paths) - 1:
            self.index += 1
        self._show_current_image()

    def save_labels(self, show_popup: bool = True) -> None:
        payload = build_label_payload(self.image_dir, self.image_paths, self.labels)
        sidecar = build_label_sidecar(self.image_dir, self.output_path, self.image_paths, self.labels)
        sidecar_path = save_labels_to_disk(payload, sidecar, self.output_path)

        counts = compute_label_counts(self.labels)
        print("\nCurrent label counts")
        print("=" * 40)
        for value in sorted(LABEL_NAMES):
            print(f"{value:>2}  {LABEL_NAMES[value]:<16} {counts[LABEL_NAMES[value]]}")
        print("=" * 40)
        print(f"Total images: {len(self.labels)}\n")

        if show_popup:
            messagebox.showinfo(
                "Saved", f"Saved labels to:\n{self.output_path}\n\nAlso wrote:\n{sidecar_path}"
            )

    def quit(self) -> None:
        self.save_labels(show_popup=False)
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir", type=Path, required=True, help="Directory containing images such as scene_000.png"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to save the .npy labels file"
    )
    parser.add_argument("--max-width", type=int, default=1000)
    parser.add_argument("--max-height", type=int, default=750)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = tk.Tk()
    app = ImageLabeler(
        root=root,
        image_dir=args.image_dir,
        output_path=args.output,
        max_width=args.max_width,
        max_height=args.max_height,
    )
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
