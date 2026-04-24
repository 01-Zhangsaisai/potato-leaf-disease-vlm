import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


# =========================
# 0. 固定随机种子
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =========================
# 1. 路径配置
# =========================
PROJECT_DIR = Path(__file__).resolve().parent

DATA_ROOT = Path(r"D:\研究生毕业论文实验\Pak-PLD-Potato")
TRAIN_DIR = DATA_ROOT / "Training"
VAL_DIR = DATA_ROOT / "Validation"
TEST_DIR = DATA_ROOT / "Testing"

RUN_NAME = "runs_resnet50_pakpld"
SAVE_DIR = PROJECT_DIR / RUN_NAME

MODELS_DIR = SAVE_DIR / "models"
LOGS_DIR = SAVE_DIR / "logs"
REPORTS_DIR = SAVE_DIR / "reports"
FIGURES_DIR = SAVE_DIR / "figures"

for d in [SAVE_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 参数配置
# =========================
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 0   # Windows 推荐 0，更稳
EARLY_STOPPING_PATIENCE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


# =========================
# 3. 检查数据目录
# =========================
for p in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not p.exists():
        raise FileNotFoundError(f"找不到目录: {p}")

print(f"Project dir : {PROJECT_DIR}")
print(f"Data root   : {DATA_ROOT}")
print(f"Device      : {DEVICE}")
print(f"Save dir    : {SAVE_DIR}")


# =========================
# 4. 数据预处理
# =========================
weights = models.ResNet50_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


# =========================
# 5. 加载数据
# =========================
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tf)
test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tf)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

class_names = train_ds.classes
num_classes = len(class_names)

if val_ds.classes != class_names or test_ds.classes != class_names:
    raise ValueError("Training / Validation / Testing 的类别文件夹名称不一致")

with open(LOGS_DIR / "classes.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("Classes:", class_names)
print(f"Train samples: {len(train_ds)}")
print(f"Val samples  : {len(val_ds)}")
print(f"Test samples : {len(test_ds)}")


# =========================
# 6. 构建模型
# =========================
model = models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)


# =========================
# 7. 训练与验证函数
# =========================
def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================
# 8. 正式训练
# =========================
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "lr": []
}

best_val_acc = 0.0
best_epoch = 0
no_improve_count = 0

for epoch in range(EPOCHS):
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]  lr={current_lr:.6f}")

    train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer=optimizer)
    val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer=None)

    scheduler.step(val_acc)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    print(
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        no_improve_count = 0
        torch.save(model.state_dict(), MODELS_DIR / "best_resnet50_pakpld.pth")
        print(">> 保存当前最佳模型")
    else:
        no_improve_count += 1

    if no_improve_count >= EARLY_STOPPING_PATIENCE:
        print(f">> Early stopping: 连续 {EARLY_STOPPING_PATIENCE} 轮验证集无提升")
        break


# 保存 history.json
with open(LOGS_DIR / "history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

# 保存 train_log.csv
train_log_df = pd.DataFrame({
    "epoch": list(range(1, len(history["train_loss"]) + 1)),
    "train_loss": history["train_loss"],
    "train_acc": history["train_acc"],
    "val_loss": history["val_loss"],
    "val_acc": history["val_acc"],
    "lr": history["lr"],
})
train_log_df.to_csv(LOGS_DIR / "train_log.csv", index=False, encoding="utf-8-sig")


# =========================
# 9. 绘制训练曲线
# =========================
epochs_range = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, history["train_loss"], marker="o", label="Train Loss")
plt.plot(epochs_range, history["val_loss"], marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet50 Loss Curve (Pak-PLD)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "loss_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, history["train_acc"], marker="o", label="Train Accuracy")
plt.plot(epochs_range, history["val_acc"], marker="o", label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet50 Accuracy Curve (Pak-PLD)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "accuracy_curve.png", dpi=300)
plt.close()


# =========================
# 10. 测试集评估
# =========================
best_model_path = MODELS_DIR / "best_resnet50_pakpld.pth"
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())


# =========================
# 11. 分类报告
# =========================
report_text = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
)

with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Classification Report (Test Set)\n")
    f.write("=" * 50 + "\n")
    f.write(report_text)

print("\n===== Test Classification Report =====")
print(report_text)


# =========================
# 12. 混淆矩阵
# =========================
cm = confusion_matrix(all_labels, all_preds)

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(REPORTS_DIR / "confusion_matrix.csv", encoding="utf-8-sig")

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix (Pak-PLD)")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
plt.close()


# =========================
# 13. 保存逐图预测结果
# =========================
# test_ds.samples: [(path, class_index), ...]
pred_rows = []
for idx, (img_path, true_idx) in enumerate(test_ds.samples):
    pred_idx = all_preds[idx]
    prob_vec = all_probs[idx]

    row = {
        "image_path": img_path,
        "true_label": class_names[true_idx],
        "pred_label": class_names[pred_idx],
        "correct": int(true_idx == pred_idx)
    }

    for i, cls_name in enumerate(class_names):
        row[f"prob_{cls_name}"] = round(float(prob_vec[i]), 6)

    pred_rows.append(row)

pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(REPORTS_DIR / "predictions.csv", index=False, encoding="utf-8-sig")


# =========================
# 14. 保存总摘要
# =========================
summary = {
    "project_dir": str(PROJECT_DIR),
    "data_root": str(DATA_ROOT),
    "save_dir": str(SAVE_DIR),
    "device": DEVICE,
    "num_classes": num_classes,
    "class_names": class_names,
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
    "test_samples": len(test_ds),
    "epochs_completed": len(history["train_loss"]),
    "best_val_acc": round(float(best_val_acc), 6),
    "best_epoch": best_epoch,
    "best_model_path": str(best_model_path)
}

with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)


print("\n训练与评估完成")
print(f"所有结果保存在: {SAVE_DIR}")
print(f"模型文件: {best_model_path}")