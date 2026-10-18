
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ==================================================
# 1. SETTINGS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_DIR
    / "PlantVillage-Dataset"
    / "data_distribution_for_SVM"
    / "train"
)

MAPPING_FILE = DATA_DIR.parent / "train_mapping.txt"
TEST_DATA_DIR = DATA_DIR.parent / "test"
TEST_MAPPING_FILE = DATA_DIR.parent / "test_mapping.txt"

OUTPUT_DIR = PROJECT_DIR / "models"
MODEL_PATH = OUTPUT_DIR / "microfarmcast_plant_disease_model.pth"
CLASSES_PATH = OUTPUT_DIR / "plant_disease_classes.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
BACKBONE_LEARNING_RATE = 0.00001
CLASSIFIER_LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==================================================
# 2. CHECK THE DATASET
# ==================================================

print("=" * 60)
print("MICROFARMCAST PLANT DISEASE MODEL TRAINING")
print("=" * 60)

print("Device:", DEVICE)
print("Dataset:", DATA_DIR)
print("Mapping file:", MAPPING_FILE)

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"\nTraining folder not found:\n{DATA_DIR}\n\n"
        "Make sure PlantVillage-Dataset is extracted "
        "inside your project folder."
    )

if not MAPPING_FILE.is_file():
    raise FileNotFoundError(
        f"\nMapping file not found:\n{MAPPING_FILE}\n\n"
        "The SVM dataset's train_mapping.txt is required "
        "to translate numbered folders into disease names."
    )

if not TEST_DATA_DIR.is_dir() or not TEST_MAPPING_FILE.is_file():
    raise FileNotFoundError(
        "The held-out test folder and test_mapping.txt are required "
        "to measure accuracy after each epoch."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. READ THE ORIGINAL DISEASE LABELS
# ==================================================

def read_label_mapping(mapping_path):
    label_by_filename = {}

    with mapping_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")

            if len(parts) < 2:
                continue

            path_parts = parts[0].replace("\\", "/").split("/")[:-1]
            disease_label = next(
                (part for part in reversed(path_parts) if "___" in part),
                None
            )

            if disease_label:
                filename = Path(parts[-1].replace("\\", "/")).name.lower()
                label_by_filename[filename] = disease_label

    if not label_by_filename:
        raise RuntimeError(
            f"Could not extract disease labels from {mapping_path.name}."
        )

    return label_by_filename


label_by_filename = read_label_mapping(MAPPING_FILE)
test_label_by_filename = read_label_mapping(TEST_MAPPING_FILE)
print("Training mapping entries read:", len(label_by_filename))
print("Test mapping entries read:", len(test_label_by_filename))


# ==================================================
# 4. LOAD ALL TRAINING IMAGES
# ==================================================

# Augmentation is applied to training images only.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

evaluation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ImageFolder discovers the numbered class directories.
dataset = datasets.ImageFolder(
    root=str(DATA_DIR),
    transform=transform
)

if len(dataset) == 0:
    raise RuntimeError(
        "No images found in the training folder. "
        "Check that the numbered folders contain images."
    )

# Translate every numbered-folder image to its real label.
resolved_samples = []
missing_images = []

for image_path, _old_label in dataset.samples:
    filename = Path(image_path).name.lower()
    disease_label = label_by_filename.get(filename)

    if disease_label is None:
        missing_images.append(image_path)
    else:
        resolved_samples.append((image_path, disease_label))

if missing_images:
    examples = "\n".join(missing_images[:5])

    raise RuntimeError(
        f"\nCould not find mapping entries for "
        f"{len(missing_images)} training images.\n"
        f"Example files:\n{examples}\n\n"
        "Check that train_mapping.txt belongs to this exact "
        "dataset folder."
    )

# Create stable class IDs using actual disease names.
class_names = sorted(
    {label for _, label in resolved_samples}
)

class_to_idx = {
    name: index
    for index, name in enumerate(class_names)
}

# Replace ImageFolder's numbered labels with disease labels.
dataset.samples = [
    (image_path, class_to_idx[disease_label])
    for image_path, disease_label in resolved_samples
]

dataset.imgs = dataset.samples
dataset.targets = [
    label for _, label in dataset.samples
]
dataset.classes = class_names
dataset.class_to_idx = class_to_idx

num_classes = len(class_names)

test_dataset = datasets.ImageFolder(
    root=str(TEST_DATA_DIR),
    transform=evaluation_transform
)
test_samples = []
missing_test_images = []

for image_path, _old_label in test_dataset.samples:
    filename = Path(image_path).name.lower()
    disease_label = test_label_by_filename.get(filename)

    if disease_label not in class_to_idx:
        missing_test_images.append(image_path)
    else:
        test_samples.append((image_path, class_to_idx[disease_label]))

if missing_test_images:
    examples = "\n".join(missing_test_images[:5])
    raise RuntimeError(
        f"Could not resolve labels for {len(missing_test_images)} test images.\n"
        f"Example files:\n{examples}"
    )

test_dataset.samples = test_samples
test_dataset.imgs = test_samples
test_dataset.targets = [label for _, label in test_samples]
test_dataset.classes = class_names
test_dataset.class_to_idx = class_to_idx

print("\nDataset successfully loaded.")
print("Total training images:", len(dataset))
print("Total disease classes:", num_classes)
print("\nDisease classes:")

for index, name in enumerate(class_names):
    print(f"  {index}: {name}")

# Save readable class labels for future predictions.
with CLASSES_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        {
            "classes": class_names,
            "class_to_idx": class_to_idx
        },
        file,
        indent=4
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# ==================================================
# 5. LOAD MOBILENETV2
# ==================================================

pretrained_loaded = False

try:
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    pretrained_loaded = True

    print("\nPretrained MobileNetV2 weights loaded.")

except Exception as error:
    print("\nCould not load pretrained weights.")
    print("Reason:", error)
    print("Using randomly initialized MobileNetV2 instead.")

    model = models.mobilenet_v2(weights=None)

# Replace the final layer for this dataset's classes.
model.classifier[1] = nn.Linear(
    model.last_channel,
    num_classes
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

if pretrained_loaded:
    optimizer = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": BACKBONE_LEARNING_RATE},
        {"params": model.classifier.parameters(), "lr": CLASSIFIER_LEARNING_RATE},
    ])
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=BACKBONE_LEARNING_RATE)


# ==================================================
# 6. TRAIN ON THE ENTIRE DATASET
# ==================================================

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        total += labels.size(0)

    epoch_loss = running_loss / total

    model.eval()
    correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = 100.0 * correct / test_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} | "
        f"Held-out accuracy: {test_accuracy:.2f}%"
    )


# ==================================================
# 7. SAVE THE MODEL
# ==================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "image_size": IMAGE_SIZE,
        "architecture": "mobilenet_v2",
        "pretrained_weights_used": pretrained_loaded
    },
    MODEL_PATH
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Images used:", len(dataset))
print("Disease classes:", num_classes)
print("Model saved to:", MODEL_PATH)
print("Class labels saved to:", CLASSES_PATH)

print(
    "\nNote: Held-out accuracy is measured on the separate "
    "test split."
)
