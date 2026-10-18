# ============================================================
# MicroFarmCast - Tomato Leaf Disease Detection
# Local PlantVillage dataset integration
# Model: MobileNetV2
# ============================================================

import json
import random
import shutil
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
SEED = 42

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "microfarmcast_tomato_model.pth"
CLASSES_PATH = MODEL_DIR / "tomato_disease_classes.json"
RESULTS_PATH = MODEL_DIR / "evaluation_results.json"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("MICROFARMCAST DISEASE DETECTION")
print("=" * 60)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Device:", device)


# ============================================================
# DATASET DISCOVERY
# ============================================================
def find_dataset_root() -> Path | None:
    candidates = [
        BASE_DIR / "PlantVillage-Dataset",
        BASE_DIR / "PlantVillage-Dataset-master",
        BASE_DIR / "PlantVillage-Dataset-main",
        BASE_DIR / "plantvillage",
        BASE_DIR / "PlantVillage",
        BASE_DIR / "dataset" / "PlantVillage-Dataset",
        BASE_DIR / "raw",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_dataset_root() -> Path:
    root = find_dataset_root()
    if root is not None:
        return root

    target_dir = BASE_DIR / "PlantVillage-Dataset"
    print("PlantVillage dataset not found. Attempting to download the official repo...")

    git_path = shutil.which("git")
    try:
        if git_path:
            subprocess.run(
                [git_path, "clone", "--depth", "1", "https://github.com/spMohanty/PlantVillage-Dataset.git", str(target_dir)],
                check=True,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return target_dir

        archive_url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
        archive_path = BASE_DIR / "PlantVillage-Dataset.zip"
        urllib.request.urlretrieve(archive_url, archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(BASE_DIR)

        for extracted_name in ["PlantVillage-Dataset-master", "PlantVillage-Dataset-main"]:
            extracted_dir = BASE_DIR / extracted_name
            if extracted_dir.exists():
                extracted_dir.rename(target_dir)
                return target_dir

        if archive_path.exists():
            archive_path.unlink()

    except Exception as exc:
        if target_dir.exists():
            return target_dir
        raise RuntimeError(
            "PlantVillage dataset not found. Clone the repo into the project folder as: "
            "PlantVillage-Dataset or make sure internet access is available for automatic download."
        ) from exc

    raise RuntimeError(
        "PlantVillage dataset not found. Clone the repo into the project folder as: "
        "PlantVillage-Dataset"
    )


def discover_tomato_samples(root: Path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    grouped = {}

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in image_extensions:
            continue

        rel_parts = file_path.relative_to(root).parts
        class_name = None
        for part in reversed(rel_parts):
            if "___" in part:
                class_name = part
                break
            if "tomato" in part.lower():
                class_name = part
                break

        if class_name is None and len(rel_parts) >= 2:
            class_name = rel_parts[-2]

        if class_name is None:
            continue

        if "tomato" not in class_name.lower() and "___" not in class_name:
            continue

        grouped.setdefault(class_name, []).append(str(file_path))

    if not grouped:
        return []

    samples = []
    for class_name, paths in grouped.items():
        for path in sorted(paths):
            samples.append((path, class_name))

    return samples


# ============================================================
# DATASET CLASS
# ============================================================
class TomatoDataset(Dataset):
    def __init__(self, samples, label_to_id, transform):
        self.samples = samples
        self.label_to_id = label_to_id
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.label_to_id[class_name]
        return image, label


print("\nLoading PlantVillage dataset...")
root = ensure_dataset_root()

print(f"Found dataset root: {root}")
all_samples = discover_tomato_samples(root)
if not all_samples:
    raise RuntimeError(
        f"No tomato images found under {root}. Check the extracted folder structure."
    )

class_names = sorted({class_name for _, class_name in all_samples})
label_to_id = {name: idx for idx, name in enumerate(class_names)}

print(f"Detected labels: {class_names}")
print(f"Total images: {len(all_samples)}")

# Split into train/test/validation
labels = [label_to_id[class_name] for _, class_name in all_samples]
train_idx, temp_idx = train_test_split(
    list(range(len(all_samples))),
    test_size=0.5,
    random_state=SEED,
    stratify=labels,
)

test_idx, val_idx = train_test_split(
    temp_idx,
    test_size=0.4,
    random_state=SEED,
    stratify=[labels[i] for i in temp_idx],
)

train_samples = [all_samples[i] for i in train_idx]
test_samples = [all_samples[i] for i in test_idx]
val_samples = [all_samples[i] for i in val_idx]

print("\nDataset split summary")
print("-" * 40)
print(f"Training: {len(train_samples)}")
print(f"Testing: {len(test_samples)}")
print(f"Validation: {len(val_samples)}")

# ============================================================
# PREPROCESSING
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = TomatoDataset(train_samples, label_to_id, transform)
test_dataset = TomatoDataset(test_samples, label_to_id, transform)
val_dataset = TomatoDataset(val_samples, label_to_id, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# MODEL
# ============================================================
print("\nLoading pretrained MobileNetV2...")
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

print("Model created successfully.")

# ============================================================
# TRAINING
# ============================================================
print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total_seen = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total_seen += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total_seen
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

print("\nTraining complete.")

# ============================================================
# EVALUATION
# ============================================================
def evaluate(loader, split_name):
    model.eval()
    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            actual.extend(labels.tolist())
            predicted.extend(preds)

    acc = accuracy_score(actual, predicted)
    print(f"\n{split_name} accuracy: {acc * 100:.2f}%")
    print(
        classification_report(
            actual,
            predicted,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )
    return acc


test_accuracy = evaluate(test_loader, "Test")
val_accuracy = evaluate(val_loader, "Validation")

# ============================================================
# SAVE OUTPUTS
# ============================================================
print("\nSaving model and metadata...")
model = model.cpu()
torch.save({"model_state_dict": model.state_dict(), "classes": class_names, "image_size": IMAGE_SIZE}, MODEL_PATH)

with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

results = {
    "dataset": "PlantVillage",
    "crop": "tomato",
    "training_percentage": 50,
    "testing_percentage": 30,
    "validation_percentage": 20,
    "classes": class_names,
    "test_accuracy": round(test_accuracy, 4),
    "validation_accuracy": round(val_accuracy, 4),
    "epochs": EPOCHS,
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nModel saved:", MODEL_PATH)
print("Classes saved:", CLASSES_PATH)
print("Results saved:", RESULTS_PATH)
print(f"Final test accuracy: {test_accuracy * 100:.2f}%")
print(f"Final validation accuracy: {val_accuracy * 100:.2f}%")
