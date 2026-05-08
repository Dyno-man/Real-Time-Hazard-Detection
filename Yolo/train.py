"""Train YOLO12n on the sidewalk hazard dataset.

Expected dataset structure by default:

sidewalk_hazard_dataset/
  annotations/
    train.json
    val.json
  images/
    train/
    val/
  class_map.json

The script converts COCO bounding boxes to YOLO labels, writes data.yaml, and
starts Ultralytics training.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO12n for sidewalk hazard detection.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset folder containing annotations/. Default: auto-detect.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Starting weights or model path. Default: newest run weights/best.pt, else ./best.pt, else ./yolo12n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs. Default: 3")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size. Default: 640")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Default: 16")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("yolo12n_sidewalk_runs"),
        help="Output directory for training runs. Default: ./yolo12n_sidewalk_runs",
    )
    parser.add_argument("--name", default="yolo12n_sidewalk_hazard", help="Training run name.")
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device value. Default: auto-select CUDA 0 if available, otherwise cpu.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=1,
        help="Save checkpoint every N epochs. Default: 1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete Ultralytics training run.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip COCO-to-YOLO label conversion and reuse existing labels/data.yaml.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation on the best weights after training.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Convert labels and write data.yaml/manifests, then exit before training.",
    )
    return parser.parse_args()


def require_path(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


def detect_dataset_root(dataset_root: Path | None) -> Path:
    if dataset_root is not None:
        return dataset_root

    default_root = Path("sidewalk_hazard_dataset")
    if (default_root / "annotations" / "train.json").exists():
        return default_root

    candidates = sorted(
        path.parent.parent
        for path in Path(".").rglob("annotations/train.json")
        if (path.parent / "val.json").exists()
    )
    if len(candidates) == 1:
        print("Auto-detected dataset root:", candidates[0])
        return candidates[0]
    if len(candidates) > 1:
        candidate_list = "\n".join(f"  {candidate}" for candidate in candidates)
        raise RuntimeError(f"Multiple dataset roots found. Pass --dataset-root:\n{candidate_list}")

    return default_root


def validate_dataset_paths(dataset_root: Path) -> tuple[Path, Path]:
    train_json = dataset_root / "annotations" / "train.json"
    val_json = dataset_root / "annotations" / "val.json"

    print("Dataset root:", dataset_root.resolve())
    require_path(dataset_root, "dataset_root does not exist")
    require_path(train_json, "Missing annotations/train.json")
    require_path(val_json, "Missing annotations/val.json")

    return train_json, val_json


def read_classes(train_json: Path) -> tuple[dict[int, int], dict[int, str]]:
    with train_json.open("r", encoding="utf-8") as f:
        train_data = json.load(f)

    categories = sorted(train_data["categories"], key=lambda item: item["id"])
    category_id_to_yolo_id = {}
    names = {}

    for yolo_id, category in enumerate(categories):
        category_id_to_yolo_id[category["id"]] = yolo_id
        names[yolo_id] = category.get("name", f"class{yolo_id}")

    print("Classes:")
    for class_id, class_name in names.items():
        print(f"  {class_id}: {class_name}")

    return category_id_to_yolo_id, names


def discover_image_dirs(dataset_root: Path, split: str) -> list[Path]:
    preferred = dataset_root / "images" / split
    image_dirs = []
    if preferred.exists():
        image_dirs.append(preferred)

    for path in sorted(Path(".").rglob(f"sidewalk_hazard_dataset/images/{split}")):
        if path.exists() and path not in image_dirs:
            image_dirs.append(path)

    if not image_dirs:
        raise FileNotFoundError(f"Missing images/{split} folder")

    print(f"{split.title()} image dirs:")
    for image_dir in image_dirs:
        print(f"  {image_dir}")

    return image_dirs


def label_path_for_image(image_path: Path) -> Path:
    parts = image_path.parts
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            return Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")
    raise ValueError(f"Image path does not contain an images directory: {image_path}")


def index_images(image_dirs: list[Path]) -> dict[str, Path]:
    image_lookup = {}
    for image_dir in image_dirs:
        for path in image_dir.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                image_lookup[path.name] = path
                image_lookup[path.stem] = path
    return image_lookup


def convert_coco_json_to_yolo(
    json_path: Path,
    image_dirs: list[Path],
    category_id_to_yolo_id: dict[int, int],
) -> None:
    print("Loading JSON:", json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("Indexing image dirs...")
    image_lookup = index_images(image_dirs)
    print("Images found in folders:", len({path.resolve() for path in image_lookup.values()}))

    images = {image["id"]: image for image in data["images"]}
    labels_by_image: dict[int, list[str]] = defaultdict(list)
    skipped_annotations = 0

    print("Converting annotations...")
    for annotation in data["annotations"]:
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")

        if image_id not in images or category_id not in category_id_to_yolo_id:
            skipped_annotations += 1
            continue

        if bbox is None or len(bbox) != 4:
            skipped_annotations += 1
            continue

        x, y, width, height = bbox
        if x is None or y is None or width is None or height is None or width <= 0 or height <= 0:
            skipped_annotations += 1
            continue

        image = images[image_id]
        image_width = image.get("width")
        image_height = image.get("height")
        if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
            skipped_annotations += 1
            continue

        x_center = max(0.0, min(1.0, (x + width / 2) / image_width))
        y_center = max(0.0, min(1.0, (y + height / 2) / image_height))
        norm_width = max(0.0, min(1.0, width / image_width))
        norm_height = max(0.0, min(1.0, height / image_height))
        yolo_class_id = category_id_to_yolo_id[category_id]

        labels_by_image[image_id].append(
            f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        )

    missing_images = 0
    labels_written = 0

    print("Writing labels...")
    for image_id, image in images.items():
        file_name = Path(image["file_name"]).name
        image_path = image_lookup.get(file_name) or image_lookup.get(Path(file_name).stem)

        if image_path is None:
            missing_images += 1
            continue

        label_path = label_path_for_image(image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = labels_by_image.get(image_id, [])
        with label_path.open("w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
        labels_written += 1

    print("Converted:", json_path)
    print("Images in JSON:", len(images))
    print("Images with label files written:", labels_written)
    print("Images missing from folder:", missing_images)
    print("Skipped bad annotations:", skipped_annotations)


def write_manifest(
    dataset_root: Path,
    split: str,
    json_path: Path,
    image_dirs: list[Path],
) -> Path:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_lookup = index_images(image_dirs)
    manifest_path = dataset_root / f"{split}.txt"
    missing_images = []
    image_paths = []

    for image in data["images"]:
        file_name = Path(image["file_name"]).name
        image_path = image_lookup.get(file_name) or image_lookup.get(Path(file_name).stem)
        if image_path is None:
            missing_images.append(file_name)
            continue
        image_paths.append(image_path.resolve())

    if missing_images:
        sample = ", ".join(missing_images[:5])
        raise RuntimeError(f"{split} is missing {len(missing_images)} images. First missing: {sample}")

    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(str(path) for path in image_paths) + "\n")

    print(f"Wrote {split} manifest:", manifest_path, f"({len(image_paths)} images)")
    return manifest_path


def write_data_yaml(
    dataset_root: Path,
    names: dict[int, str],
    train_manifest: Path,
    val_manifest: Path,
) -> Path:
    data_yaml_path = dataset_root / "data.yaml"
    data_yaml = {
        "path": str(dataset_root.resolve()),
        "train": str(train_manifest.resolve()),
        "val": str(val_manifest.resolve()),
        "names": names,
    }

    with data_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print("Wrote data yaml:", data_yaml_path)
    return data_yaml_path


def count_manifest_images(manifest_path: Path) -> int:
    with manifest_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def verify_converted_dataset(train_manifest: Path, val_manifest: Path) -> None:
    train_image_count = count_manifest_images(train_manifest)
    val_image_count = count_manifest_images(val_manifest)
    train_label_count = 0
    val_label_count = 0

    for manifest_path, split in ((train_manifest, "train"), (val_manifest, "val")):
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                image_path = Path(line.strip())
                if not line.strip():
                    continue
                if label_path_for_image(image_path).exists():
                    if split == "train":
                        train_label_count += 1
                    else:
                        val_label_count += 1

    print("Train images:", train_image_count)
    print("Train labels:", train_label_count)
    print("Val images:", val_image_count)
    print("Val labels:", val_label_count)

    if train_image_count == 0:
        raise RuntimeError("No training images found.")
    if val_image_count == 0:
        raise RuntimeError("No validation images found.")
    if train_label_count == 0:
        raise RuntimeError("No training labels were created.")
    if val_label_count == 0:
        raise RuntimeError("No validation labels were created.")


def choose_device(device: str | None):
    if device is not None:
        return device

    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def choose_model(model: str | None) -> str:
    if model is not None:
        return model

    run_best_weights = sorted(
        Path(".").glob("runs/**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if run_best_weights:
        return str(run_best_weights[0])

    for candidate in (Path("best.pt"), Path("yolo12n.pt")):
        if candidate.exists():
            return str(candidate)

    return "yolo12n.pt"


def train_model(args: argparse.Namespace, data_yaml_path: Path):
    from ultralytics import YOLO

    training_device = choose_device(args.device)
    print("Training device:", training_device)

    model_path = choose_model(args.model)

    model = YOLO(model_path)
    return model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=training_device,
        project=str(args.project),
        name=args.name,
        save=True,
        save_period=args.save_period,
        resume=args.resume,
    )


def validate_best_model(args: argparse.Namespace, data_yaml_path: Path) -> None:
    from ultralytics import YOLO

    training_device = choose_device(args.device)
    best_model_path = args.project / args.name / "weights" / "best.pt"
    require_path(best_model_path, "Best model was not found")

    best_model = YOLO(str(best_model_path))
    best_model.val(data=str(data_yaml_path), imgsz=args.imgsz, device=training_device)


def main() -> None:
    args = parse_args()
    dataset_root = detect_dataset_root(args.dataset_root)
    train_json, val_json = validate_dataset_paths(dataset_root)
    train_image_dirs = discover_image_dirs(dataset_root, "train")
    val_image_dirs = discover_image_dirs(dataset_root, "val")

    if args.skip_convert:
        data_yaml_path = dataset_root / "data.yaml"
        require_path(data_yaml_path, "Missing data.yaml")
        train_manifest = dataset_root / "train.txt"
        val_manifest = dataset_root / "val.txt"
        require_path(train_manifest, "Missing train.txt")
        require_path(val_manifest, "Missing val.txt")
    else:
        category_id_to_yolo_id, names = read_classes(train_json)
        convert_coco_json_to_yolo(
            train_json,
            train_image_dirs,
            category_id_to_yolo_id,
        )
        convert_coco_json_to_yolo(
            val_json,
            val_image_dirs,
            category_id_to_yolo_id,
        )
        train_manifest = write_manifest(dataset_root, "train", train_json, train_image_dirs)
        val_manifest = write_manifest(dataset_root, "val", val_json, val_image_dirs)
        data_yaml_path = write_data_yaml(dataset_root, names, train_manifest, val_manifest)

    verify_converted_dataset(train_manifest, val_manifest)
    if args.prepare_only:
        print("Dataset preparation complete. Skipping training because --prepare-only was set.")
        return

    train_model(args, data_yaml_path)

    if args.validate:
        validate_best_model(args, data_yaml_path)


if __name__ == "__main__":
    main()
