from pathlib import Path
import json
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

# CONFIG

WOTR_ROOT = Path("/Users/collintucker/Desktop/Machine Vision/DatasetTest/extracted_in_container/WOTR")
POTHOLE_ROOT = Path("/Users/collintucker/.cache/kagglehub/datasets/sabidrahman/pothole-cracks-and-openmanhole/versions/2")

OUTPUT_ROOT = Path("./sidewalk_hazard_dataset")

VAL_RATIO = 0.2
RANDOM_SEED = 42

CLASSES = (
    "person",
    "bicycle",
    "car",
    "traffic_cone",
    "pothole",
    "open_manhole",
    "pole",
    "sign",
    "trash_can",
    "roadblock",
    "tree",
    "animal",
    "fire_hydrant",
    "crosswalk",
    "tactile_paving",
    "surface_damage",
)

CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASSES)}

GLOBAL_SYNONYMS = {
    # WOTR
    "person": "person",
    "dog": "animal",

    "bicycle": "bicycle",
    "motorcycle": "bicycle",
    "tricycle": "bicycle",

    "car": "car",
    "truck": "car",
    "bus": "car",

    "reflective_cone": "traffic_cone",
    "warning_column": "traffic_cone",

    "blind_road": "tactile_paving",
    "ashcan": "trash_can",
    "roadblock": "roadblock",
    "crosswalk": "crosswalk",
    "pole": "pole",
    "sign": "sign",
    "tree": "tree",
    "fire_hydrant": "fire_hydrant",

    # Drop traffic lights for now
    "red_light": None,
    "green_light": None,

    # Pothole dataset
    "pothole": "pothole",
    "open_manhole": "open_manhole",
    "crack": "surface_damage",
    "cracks": "surface_damage",
}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# HELPERS


def reset_output_dir():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)

    (OUTPUT_ROOT / "annotations").mkdir(parents=True, exist_ok=True)


def find_image_for_xml(xml_path: Path):
    xml_dir = xml_path.parent

    try:
        root = ET.parse(xml_path).getroot()
        filename_node = root.find("filename")
        if filename_node is not None and filename_node.text:
            candidate = xml_dir / filename_node.text
            if candidate.exists():
                return candidate

            # Search nearby if XML stores only filename but image is elsewhere
            matches = list(xml_path.parents[0].glob(f"**/{filename_node.text}"))
            if matches:
                return matches[0]
    except Exception:
        pass

    stem = xml_path.stem

    search_roots = [
        xml_path.parent,
        xml_path.parent.parent,
        xml_path.parent.parent.parent if xml_path.parent.parent else xml_path.parent,
    ]

    for root in search_roots:
        if root.exists():
            for ext in IMAGE_EXTENSIONS:
                matches = list(root.glob(f"**/{stem}{ext}"))
                if matches:
                    return matches[0]

    return None


def parse_voc_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()

    image_path = find_image_for_xml(xml_path)
    if image_path is None or not image_path.exists():
        return None

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

    objects = []

    for obj in root.findall("object"):
        raw_name = obj.find("name").text.strip()
        mapped_name = GLOBAL_SYNONYMS.get(raw_name)

        if mapped_name is None:
            continue

        if mapped_name not in CLASS_TO_ID:
            print(f"Skipping unknown mapped class: {raw_name} -> {mapped_name}")
            continue

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))

        box_width = xmax - xmin
        box_height = ymax - ymin

        if box_width <= 1 or box_height <= 1:
            continue

        objects.append({
            "class_name": mapped_name,
            "bbox": [xmin, ymin, box_width, box_height],
            "area": box_width * box_height,
        })

    if len(objects) == 0:
        return None

    return {
        "image_path": image_path,
        "width": width,
        "height": height,
        "objects": objects,
    }


def collect_voc_samples(dataset_root: Path):
    xml_files = list(dataset_root.rglob("*.xml"))
    samples = []

    for i, xml_path in enumerate(xml_files):
        if i % 500 == 0:
            print(f"Processed {i}/{len(xml_files)} XML files")
        parsed = parse_voc_xml(xml_path)
        if parsed is not None:
            samples.append(parsed)

    return samples


def build_coco(samples, split_name):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": CLASS_TO_ID[class_name],
                "name": class_name,
                "supercategory": "sidewalk_hazard",
            }
            for class_name in CLASSES
        ],
    }

    annotation_id = 1

    for image_id, sample in enumerate(samples, start=1):
        src_path = sample["image_path"]
        ext = src_path.suffix.lower()
        new_filename = f"{split_name}_{image_id:06d}{ext}"
        dst_path = OUTPUT_ROOT / "images" / split_name / new_filename

        shutil.copy2(src_path, dst_path)

        coco["images"].append({
            "id": image_id,
            "file_name": new_filename,
            "width": sample["width"],
            "height": sample["height"],
        })

        for obj in sample["objects"]:
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": CLASS_TO_ID[obj["class_name"]],
                "bbox": obj["bbox"],
                "area": obj["area"],
                "iscrowd": 0,
            })
            annotation_id += 1

    return coco


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# MAIN

reset_output_dir()

print("Collecting WOTR VOC samples...")
wotr_samples = collect_voc_samples(WOTR_ROOT)
print(f"WOTR usable samples: {len(wotr_samples)}")

print("Collecting pothole/manhole/crack VOC samples...")
pothole_samples = collect_voc_samples(POTHOLE_ROOT)
print(f"Pothole dataset usable samples: {len(pothole_samples)}")

all_samples = wotr_samples + pothole_samples
print(f"Total usable samples: {len(all_samples)}")

random.seed(RANDOM_SEED)
random.shuffle(all_samples)

val_count = int(len(all_samples) * VAL_RATIO)
val_samples = all_samples[:val_count]
train_samples = all_samples[val_count:]

print(f"Train samples: {len(train_samples)}")
print(f"Val samples: {len(val_samples)}")

train_coco = build_coco(train_samples, "train")
val_coco = build_coco(val_samples, "val")

save_json(train_coco, OUTPUT_ROOT / "annotations" / "train.json")
save_json(val_coco, OUTPUT_ROOT / "annotations" / "val.json")

save_json(
    {
        "classes": CLASSES,
        "class_to_id": CLASS_TO_ID,
        "synonyms": GLOBAL_SYNONYMS,
        "notes": {
            "red_light": "dropped",
            "green_light": "dropped",
            "crack/cracks": "mapped to surface_damage",
            "dog": "mapped to animal",
            "truck/bus": "mapped to car",
            "motorcycle/tricycle": "mapped to bicycle",
        },
    },
    OUTPUT_ROOT / "class_map.json"
)

print("\nDone.")
print(f"Dataset saved to: {OUTPUT_ROOT.resolve()}")