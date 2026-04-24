import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# =========================
# 1. 基础配置
# =========================
PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = PROJECT_DIR / "runs_efficientnetb3_pakpld" / "models" / "best_efficientnetb3_pakpld.pth"
DEFAULT_CLASSES_PATH = PROJECT_DIR / "runs_efficientnetb3_pakpld" / "logs" / "classes.json"
DEFAULT_SAVE_DIR = PROJECT_DIR / "demo_outputs"

DEFAULT_CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 2. 工具函数
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict], path: Path):
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================
# 3. 加载类别名称
# =========================
def load_class_names(classes_path: Path) -> List[str]:
    if classes_path.exists():
        class_names = load_json(classes_path)
        if not isinstance(class_names, list) or len(class_names) == 0:
            raise ValueError(f"classes.json 格式不正确: {classes_path}")
        return class_names
    return DEFAULT_CLASS_NAMES


# =========================
# 4. 构建模型
# =========================
def build_efficientnetb3(num_classes: int):
    weights = models.EfficientNet_B3_Weights.DEFAULT
    model = models.efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model, weights


def build_transform(weights):
    mean = weights.transforms().mean
    std = weights.transforms().std
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_state_dict_safely(model_path: Path):
    try:
        # 新版本 torch 推荐写法
        return torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        # 兼容旧版本 torch
        return torch.load(model_path, map_location=DEVICE)


def load_model(model_path: Path, class_names: List[str]):
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    model, weights = build_efficientnetb3(num_classes=len(class_names))
    state_dict = load_state_dict_safely(model_path)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    transform = build_transform(weights)
    return model, transform


# =========================
# 5. 采集输入图片
# =========================
def collect_images(input_path: Path, recursive: bool = False, max_images: int = 0) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"输入文件不是支持的图片格式: {input_path}")
        return [input_path]

    if input_path.is_dir():
        if recursive:
            images = [p for p in input_path.rglob("*") if is_image_file(p)]
        else:
            images = [p for p in input_path.iterdir() if is_image_file(p)]

        images = sorted(images)
        if max_images > 0:
            images = images[:max_images]

        if not images:
            raise ValueError(f"该目录下未找到可用图片: {input_path}")

        return images

    raise ValueError(f"无法识别的输入路径类型: {input_path}")


# =========================
# 6. 预测逻辑
# =========================
def predict_image(model, transform, image_path: Path, class_names: List[str]) -> Dict:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx].item())

    prob_dict = {
        class_names[i]: float(probs[i].item())
        for i in range(len(class_names))
    }

    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    second_confidence = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin = confidence - second_confidence

    return {
        "image_path": str(image_path),
        "device": DEVICE,
        "predicted_class": pred_class,
        "confidence": confidence,
        "second_confidence": second_confidence,
        "margin": margin,
        "probabilities": prob_dict
    }


# =========================
# 7. 诊断文本生成
# =========================
def get_prediction_status(confidence: float, margin: float, low_threshold: float, medium_threshold: float, margin_threshold: float):
    if confidence < low_threshold or margin < margin_threshold:
        return "uncertain"
    if confidence < medium_threshold:
        return "cautious"
    return "confident"


def generate_diagnosis_text(
    result: Dict,
    low_threshold: float = 0.60,
    medium_threshold: float = 0.85,
    margin_threshold: float = 0.15
) -> Tuple[str, str]:
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    margin = result["margin"]
    probs = result["probabilities"]

    conf_pct = confidence * 100
    margin_pct = margin * 100
    eb_pct = probs.get("Early_Blight", 0.0) * 100
    he_pct = probs.get("Healthy", 0.0) * 100
    lb_pct = probs.get("Late_Blight", 0.0) * 100

    class_name_zh = {
        "Healthy": "健康",
        "Early_Blight": "早疫病",
        "Late_Blight": "晚疫病",
    }
    pred_class_zh = class_name_zh.get(pred_class, pred_class)

    status = get_prediction_status(confidence, margin, low_threshold, medium_threshold, margin_threshold)

    if status == "uncertain":
        text = (
            f"诊断结果：当前模型对该图像的判断不够稳定。"
            f"最高类别为“{pred_class_zh}”，置信度为 {conf_pct:.2f}%，"
            f"与第二候选类别的差距为 {margin_pct:.2f}%。"
            f"该图像可能不是标准马铃薯叶片图像，或者图像质量、拍摄角度、背景条件与训练数据差异较大。"
            f"因此不建议直接给出明确病害结论。"
            f"各类别概率分别为：健康 {he_pct:.2f}%，早疫病 {eb_pct:.2f}%，晚疫病 {lb_pct:.2f}%。"
        )
        return status, text

    if status == "cautious":
        caution = "结果需要谨慎参考。"
    else:
        caution = "结果可信度很高。"

    if pred_class == "Healthy":
        diagnosis = (
            f"诊断结果：这片叶子判定为健康状态。"
            f"模型置信度为 {conf_pct:.2f}%，{caution}"
            f"当前未检测到明显的早疫病或晚疫病特征。"
        )
        suggestion = "建议继续保持观察，并结合后续图像采集进行周期性监测。"

    elif pred_class == "Early_Blight":
        diagnosis = (
            f"诊断结果：这片叶子疑似患有早疫病。"
            f"模型置信度为 {conf_pct:.2f}%，{caution}"
            f"该结果通常对应叶面出现褐色坏死斑点或局部病斑扩展等症状。"
        )
        suggestion = "建议结合原始叶片图像进行人工复核，并继续观察病斑扩展情况。"

    elif pred_class == "Late_Blight":
        diagnosis = (
            f"诊断结果：这片叶子疑似患有晚疫病。"
            f"模型置信度为 {conf_pct:.2f}%，{caution}"
            f"该结果通常对应较大面积病斑、组织受损或边缘坏死等表现。"
        )
        suggestion = "建议尽快结合实际植株状态进行复核，并关注病害进一步传播风险。"

    else:
        diagnosis = (
            f"诊断结果：模型当前预测类别为 {pred_class_zh}，"
            f"置信度为 {conf_pct:.2f}%，{caution}"
        )
        suggestion = "建议结合图像内容进一步判断。"

    details = (
        f"各类别概率分别为：健康 {he_pct:.2f}%，"
        f"早疫病 {eb_pct:.2f}%，"
        f"晚疫病 {lb_pct:.2f}%。"
    )

    return status, diagnosis + details + suggestion


# =========================
# 8. 打印输出
# =========================
def print_single_result(result: Dict, diagnosis_text: str, topk: int = 3):
    print("\n===== 分类结果 =====")
    print(f"图片路径: {result['image_path']}")
    print(f"运行设备: {result['device']}")
    print(f"预测类别: {result['predicted_class']}")
    print(f"预测置信度: {result['confidence']:.6f}")
    print(f"与第二候选差距: {result['margin']:.6f}")

    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)

    print("\n===== Top 概率 =====")
    for cls_name, prob in sorted_probs[:topk]:
        print(f"{cls_name:<15}: {prob:.6f}")

    print("\n===== 中文诊断输出 =====")
    print(diagnosis_text)


def summarize_batch(rows: List[Dict]):
    total = len(rows)
    success_rows = [r for r in rows if r.get("status") != "error"]
    error_rows = [r for r in rows if r.get("status") == "error"]

    count_by_class = {}
    for r in success_rows:
        pred = r.get("predicted_class", "Unknown")
        count_by_class[pred] = count_by_class.get(pred, 0) + 1

    print("\n===== 批量处理完成 =====")
    print(f"总图片数: {total}")
    print(f"成功处理: {len(success_rows)}")
    print(f"失败数量: {len(error_rows)}")

    if count_by_class:
        print("\n预测类别统计:")
        for cls_name, cnt in sorted(count_by_class.items()):
            print(f"  {cls_name:<15}: {cnt}")


# =========================
# 9. 主程序
# =========================
def main():
    parser = argparse.ArgumentParser(description="Potato leaf diagnosis main entry")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图片路径，既支持单张图片，也支持整个文件夹"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="当输入为文件夹时，是否递归读取子文件夹"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="批量模式下最多处理多少张图片，0 表示不限制"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="模型权重路径"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=str(DEFAULT_CLASSES_PATH),
        help="classes.json 路径"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(DEFAULT_SAVE_DIR),
        help="输出结果保存目录"
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=0.60,
        help="低置信度阈值，低于该值将提示结果不稳定"
    )
    parser.add_argument(
        "--medium_threshold",
        type=float,
        default=0.85,
        help="中等置信度阈值"
    )
    parser.add_argument(
        "--margin_threshold",
        type=float,
        default=0.15,
        help="第一候选与第二候选的最小差距阈值"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="单图模式下显示前 k 个概率"
    )

    args = parser.parse_args()

    input_path = Path(args.image)
    model_path = Path(args.model)
    classes_path = Path(args.classes)
    save_dir = Path(args.save_dir)

    ensure_dir(save_dir)

    class_names = load_class_names(classes_path)
    model, transform = load_model(model_path, class_names)

    image_paths = collect_images(
        input_path=input_path,
        recursive=args.recursive,
        max_images=args.max_images
    )

    # 单图模式
    if len(image_paths) == 1 and image_paths[0].is_file():
        image_path = image_paths[0]
        try:
            result = predict_image(model, transform, image_path, class_names)
            status, diagnosis_text = generate_diagnosis_text(
                result,
                low_threshold=args.low_threshold,
                medium_threshold=args.medium_threshold,
                margin_threshold=args.margin_threshold
            )
            result["status"] = status
            result["diagnosis_text"] = diagnosis_text

            print_single_result(result, diagnosis_text, topk=args.topk)

            output_json = save_dir / f"{safe_stem(image_path)}_result.json"
            save_json(result, output_json)
            print(f"\n结果已保存到: {output_json}")

        except (UnidentifiedImageError, OSError) as e:
            print(f"\n图片读取失败: {image_path}")
            print(f"错误信息: {e}")
        return

    # 批量模式
    batch_rows = []
    batch_json = []

    print(f"\n检测到文件夹输入，共找到 {len(image_paths)} 张图片，开始批量处理...")

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            result = predict_image(model, transform, image_path, class_names)
            status, diagnosis_text = generate_diagnosis_text(
                result,
                low_threshold=args.low_threshold,
                medium_threshold=args.medium_threshold,
                margin_threshold=args.margin_threshold
            )

            row = {
                "index": idx,
                "image_path": str(image_path),
                "predicted_class": result["predicted_class"],
                "confidence": round(result["confidence"], 6),
                "margin": round(result["margin"], 6),
                "status": status,
                "prob_Healthy": round(result["probabilities"].get("Healthy", 0.0), 6),
                "prob_Early_Blight": round(result["probabilities"].get("Early_Blight", 0.0), 6),
                "prob_Late_Blight": round(result["probabilities"].get("Late_Blight", 0.0), 6),
                "diagnosis_text": diagnosis_text,
            }
            batch_rows.append(row)

            result["status"] = status
            result["diagnosis_text"] = diagnosis_text
            batch_json.append(result)

            print(f"[{idx}/{len(image_paths)}] 完成: {image_path.name} -> {result['predicted_class']} ({result['confidence']:.4f})")

        except (UnidentifiedImageError, OSError, RuntimeError, ValueError) as e:
            error_row = {
                "index": idx,
                "image_path": str(image_path),
                "predicted_class": "",
                "confidence": "",
                "margin": "",
                "status": "error",
                "prob_Healthy": "",
                "prob_Early_Blight": "",
                "prob_Late_Blight": "",
                "diagnosis_text": f"处理失败: {e}",
            }
            batch_rows.append(error_row)
            batch_json.append(error_row)
            print(f"[{idx}/{len(image_paths)}] 失败: {image_path.name} -> {e}")

    csv_path = save_dir / "batch_results.csv"
    json_path = save_dir / "batch_results.json"

    save_csv(batch_rows, csv_path)
    save_json(batch_json, json_path)

    summarize_batch(batch_rows)
    print(f"\nCSV 已保存到: {csv_path}")
    print(f"JSON 已保存到: {json_path}")


if __name__ == "__main__":
    main()