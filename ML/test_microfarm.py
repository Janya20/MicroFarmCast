
from pathlib import Path
import json
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score


# ==================================================
# 1. SETTINGS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

TEST_DIR = (
    PROJECT_DIR
    / "PlantVillage-Dataset"
    / "data_distribution_for_SVM"
    / "test"
)

TEST_MAPPING = TEST_DIR.parent / "test_mapping.txt"

MODEL_PATH = (
    PROJECT_DIR / "models"
    / "microfarmcast_plant_disease_model.pth"
)

CLASSES_PATH = (
    PROJECT_DIR / "models"
    / "plant_disease_classes.json"
)

BATCH_SIZE = 32
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# 2. CHECK FILES
# ==================================================

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

if not TEST_MAPPING.is_file():
    raise FileNotFoundError(
        f"Test mapping file not found: {TEST_MAPPING}"
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Train the model first."
    )

if not CLASSES_PATH.is_file():
    raise FileNotFoundError(
        f"Class labels file not found: {CLASSES_PATH}"
    )


# ==================================================
# 3. LOAD CLASS LABELS
# ==================================================

with CLASSES_PATH.open("r", encoding="utf-8") as file:
    class_info = json.load(file)

class_names = class_info["classes"]
class_to_idx = class_info["class_to_idx"]

# Ensure class IDs are integers.
class_to_idx = {
    name: int(index)
    for name, index in class_to_idx.items()
}

num_classes = len(class_names)

print("Device:", DEVICE)
print("Number of classes:", num_classes)


# ==================================================
# 4. READ TEST LABEL MAPPING
# ==================================================

test_labels_by_filename = {}

with TEST_MAPPING.open("r", encoding="utf-8-sig") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        columns = line.split("\t")

        if len(columns) < 2:
            continue

        original_path = columns[0].strip()
        copied_path = columns[-1].strip()

        original_parts = (
            original_path.replace("\\", "/").split("/")[:-1]
        )

        disease_name = next(
            (
                part for part in reversed(original_parts)
                if "___" in part
            ),
            None
        )

        if disease_name is None:
            continue

        filename = Path(
            copied_path.replace("\\", "/")
        ).name.lower()

        test_labels_by_filename[filename] = disease_name


# ==================================================
# 5. LOAD TEST IMAGES WITHOUT AUGMENTATION
# ==================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

folder_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=test_transform
)

if len(folder_dataset) == 0:
    raise RuntimeError("No test images found.")

test_samples = []
missing = []

for image_path, _ in folder_dataset.samples:
    filename = Path(image_path).name.lower()
    disease_name = test_labels_by_filename.get(filename)

    if disease_name is None:
        missing.append(image_path)
        continue

    if disease_name not in class_to_idx:
        raise ValueError(
            f"Test class '{disease_name}' is missing "
            "from the training class list."
        )

    test_samples.append(
        (image_path, class_to_idx[disease_name])
    )

if missing:
    raise RuntimeError(
        f"{len(missing)} test images have no mapping entry. "
        "Check test_mapping.txt."
    )

if not test_samples:
    raise RuntimeError("No mapped test images found.")


class TestImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        return self.transform(image), label


test_dataset = TestImageDataset(
    test_samples,
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# ==================================================
# 6. LOAD THE TRAINED MODEL
# ==================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if checkpoint.get("architecture") != "mobilenet_v2":
    raise ValueError("The saved model is not MobileNetV2.")

if checkpoint["num_classes"] != num_classes:
    raise ValueError(
        "The model class count does not match the class file."
    )

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.last_channel,
    num_classes
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()


# ==================================================
# 7. EVALUATE ON UNSEEN TEST IMAGES
# ==================================================

actual_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        predicted_labels.extend(predictions)
        actual_labels.extend(labels.tolist())


# ==================================================
# 8. PRINT RESULTS
# ==================================================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

print("\n" + "=" * 55)
print("MICROFARMCAST TEST RESULTS")
print("=" * 55)

print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"Correct predictions: {sum(a == p for a, p in zip(actual_labels, predicted_labels))}")
print(f"Total test images: {len(actual_labels)}")

print("\nClassification report(.venv) PS D:\Mini project\Datathon project> git status
fatal: not a git repository (or any of the parent directories): .git
:\n")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )
)
