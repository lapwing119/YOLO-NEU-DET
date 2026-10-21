from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

RAW = Path("raw/NEU-DET")
OUT = Path("dataset")
SEED = 42

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

if not RAW.exists():
    raise SystemExit(f"Raw dataset folder not found: {RAW.resolve()}")

if any(p.is_file() for p in OUT.rglob("*")):
    raise SystemExit(
        f"Output folder is not empty: {OUT.resolve()}\n"
        "Move existing dataset files elsewhere before rebuilding."
    )

image_by_stem = {}
for p in RAW.rglob("*"):
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        key = p.stem.lower()
        if key in image_by_stem:
            raise SystemExit(f"Duplicate image filename stem: {key}")
        image_by_stem[key] = p

records = []
for xml_path in sorted(RAW.rglob("*.xml")):
    xml_root = ET.parse(xml_path).getroot()
    stem = xml_path.stem.lower()

    if stem not in image_by_stem:
        raise SystemExit(f"No matching image for XML: {xml_path}")

    source_split = image_by_stem[stem].relative_to(RAW).parts[0].lower()
    if source_split not in {"train", "validation"}:
        raise SystemExit(f"Unexpected image folder: {image_by_stem[stem]}")

    width = int(xml_root.findtext("size/width"))
    height = int(xml_root.findtext("size/height"))
    if width <= 0 or height <= 0:
        raise SystemExit(f"Invalid image size in XML: {xml_path}")

    labels = []
    for obj in xml_root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if class_name not in CLASS_TO_ID:
            raise SystemExit(f"Unknown class '{class_name}' in {xml_path}")

        box = obj.find("bndbox")
        if box is None:
            raise SystemExit(f"Missing bndbox in {xml_path}")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise SystemExit(f"Invalid bounding box in {xml_path}: {xmin,ymin,xmax,ymax}")

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        labels.append(
            f"{CLASS_TO_ID[class_name]} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    if not labels:
        raise SystemExit(f"No objects found in XML: {xml_path}")

    records.append({
        "stem": stem,
        "image": image_by_stem[stem],
        "source_split": source_split,
        "class_id": int(labels[0].split()[0]),
        "label_text": "\n".join(labels) + "\n",
    })

train_records = [r for r in records if r["source_split"] == "train"]
validation_records = [r for r in records if r["source_split"] == "validation"]

if len(train_records) != 1440 or len(validation_records) != 360:
    raise SystemExit(
        f"Unexpected source split sizes: train={len(train_records)}, "
        f"validation={len(validation_records)}; expected 1440 and 360."
    )

rng = random.Random(SEED)
val_records = []
test_records = []

for class_id in range(len(CLASSES)):
    group = [r for r in validation_records if r["class_id"] == class_id]
    if len(group) != 60:
        raise SystemExit(
            f"Expected 60 validation images for class {class_id}, found {len(group)}"
        )
    rng.shuffle(group)
    val_records.extend(group[:30])
    test_records.extend(group[30:])

splits = {
    "train": train_records,
    "val": val_records,
    "test": test_records,
}

for split, split_records in splits.items():
    image_dir = OUT / "images" / split
    label_dir = OUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for r in split_records:
        shutil.copy2(r["image"], image_dir / r["image"].name)
        (label_dir / f"{r['stem']}.txt").write_text(
            r["label_text"], encoding="utf-8"
        )

yaml_text = (
    "path: C:/wlkq/YOLO-NEU-DET/dataset\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
)
(OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

print("Dataset conversion complete.")
for split, split_records in splits.items():
    print(f"{split}: {len(split_records)} images")
print(f"Classes: {CLASSES}")
print(f"YAML: {(OUT / 'data.yaml').resolve()}")
