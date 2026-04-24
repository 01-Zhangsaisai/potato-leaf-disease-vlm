import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# =========================
# 1. 项目路径配置
# =========================
PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_CNN_MODEL_PATH = (
    PROJECT_DIR
    / "runs_efficientnetb3_pakpld"
    / "models"
    / "best_efficientnetb3_pakpld.pth"
)

DEFAULT_CLASSES_PATH = (
    PROJECT_DIR
    / "runs_efficientnetb3_pakpld"
    / "logs"
    / "classes.json"
)

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "demo_outputs_vlm"

DEFAULT_CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

CNN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 2. 固定术语表
# =========================
CLASS_TERMS = {
    "Healthy": {
        "zh": "健康叶片",
        "ru": "здоровый лист",
        "ru_full": "здоровый лист картофеля",
    },
    "Early_Blight": {
        "zh": "早疫病",
        "ru": "альтернариоз",
        "ru_full": "альтернариоз картофеля, также называемый ранней пятнистостью",
    },
    "Late_Blight": {
        "zh": "晚疫病",
        "ru": "фитофтороз",
        "ru_full": "фитофтороз картофеля",
    },
}


# =========================
# 3. 基础工具函数
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_stem(path: Path) -> str:
    return (
        path.stem
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(text: str, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


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
            raise ValueError(f"该文件夹中未找到图片: {input_path}")

        return images

    raise ValueError(f"无法识别输入路径类型: {input_path}")


# =========================
# 4. 类别读取
# =========================
def load_class_names(classes_path: Path) -> List[str]:
    if classes_path.exists():
        class_names = load_json(classes_path)

        if not isinstance(class_names, list) or len(class_names) == 0:
            raise ValueError(f"classes.json 格式不正确: {classes_path}")

        return class_names

    return DEFAULT_CLASS_NAMES


# =========================
# 5. EfficientNetB3 分类模型
# =========================
def build_efficientnetb3(num_classes: int):
    weights = models.EfficientNet_B3_Weights.DEFAULT
    model = models.efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model, weights


def build_cnn_transform(weights):
    mean = weights.transforms().mean
    std = weights.transforms().std

    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_cnn_model(model_path: Path, class_names: List[str]):
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 CNN 模型文件: {model_path}")

    model, weights = build_efficientnetb3(num_classes=len(class_names))

    try:
        state_dict = torch.load(model_path, map_location=CNN_DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=CNN_DEVICE)

    model.load_state_dict(state_dict)
    model = model.to(CNN_DEVICE)
    model.eval()

    transform = build_cnn_transform(weights)
    return model, transform


def predict_with_cnn(
    image_path: Path,
    model,
    transform,
    class_names: List[str]
) -> Dict:
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(CNN_DEVICE)

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

    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    second_class = sorted_probs[1][0] if len(sorted_probs) > 1 else ""
    second_confidence = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin = confidence - second_confidence

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "second_class": second_class,
        "second_confidence": second_confidence,
        "margin": margin,
        "probabilities": prob_dict,
    }


# =========================
# 6. 诊断风险等级
# =========================
def confidence_status(confidence: float, margin: float) -> str:
    if confidence < 0.60 or margin < 0.15:
        return "uncertain"
    if confidence < 0.85:
        return "cautious"
    return "confident"


# =========================
# 7. 构造 Qwen2.5-VL Prompt
# =========================
def build_vlm_prompt(
    cnn_result: Dict,
    language: str = "zh",
    strict_retry: bool = False
) -> str:
    pred_class = cnn_result["predicted_class"]
    confidence = cnn_result["confidence"]
    margin = cnn_result["margin"]
    probs = cnn_result["probabilities"]

    zh_name = CLASS_TERMS.get(pred_class, {}).get("zh", pred_class)
    ru_name = CLASS_TERMS.get(pred_class, {}).get("ru", pred_class)
    ru_full = CLASS_TERMS.get(pred_class, {}).get("ru_full", pred_class)

    prob_lines = "\n".join([
        f"- {cls}: {prob:.6f}"
        for cls, prob in probs.items()
    ])

    status = confidence_status(confidence, margin)

    if language == "ru":
        retry_text = ""
        if strict_retry:
            retry_text = """
Предыдущий ответ содержал неправильный язык или неточную терминологию.
Повтори ответ строго только на русском языке.
Не используй китайские иероглифы.
Не используй английские названия классов в основном тексте, кроме блока с техническими вероятностями.
"""

        prompt = f"""
Ты — ассистент для интерпретации результатов распознавания заболеваний листьев картофеля.

{retry_text}

Тебе передано изображение листа картофеля и результат CNN-классификатора EfficientNetB3.

Технический результат CNN:
- Predicted class: {pred_class}
- Correct Russian term: {ru_full}
- Confidence: {confidence:.6f}
- Margin between top-1 and top-2 classes: {margin:.6f}
- Confidence status: {status}
- Class probabilities:
{prob_lines}

Фиксированный словарь терминов:
- Healthy = здоровый лист
- Early_Blight = альтернариоз / ранняя пятнистость
- Late_Blight = фитофтороз

Строгие правила:
1. Ответ должен быть только на русском языке.
2. Запрещено использовать китайский язык.
3. Не переводи Late_Blight как «ранняя плесень».
4. Late_Blight всегда формулируй как «фитофтороз».
5. Early_Blight всегда формулируй как «альтернариоз» или «ранняя пятнистость».
6. Healthy всегда формулируй как «здоровый лист».
7. Не называй фитофтороз просто грибковым заболеванием.
8. Если нужно упомянуть патоген, формулируй осторожно: «заболевание связано с Phytophthora infestans».
9. Не придумывай визуальные признаки, если они плохо видны.
10. CNN является основным классификационным модулем; VLM только объясняет результат.
11. Не утверждай диагноз как абсолютно точный.
12. В конце обязательно укажи, что результат является вспомогательным и требует проверки специалистом.

Сформируй ответ строго в такой структуре:

Диагностическое заключение:
...

Визуальные признаки на изображении:
...

Ограничения и необходимость проверки:
...
"""
        return prompt.strip()

    prompt = f"""
你是一个用于解释马铃薯叶片病害识别结果的视觉语言诊断助手。

你会接收到一张马铃薯叶片图像，以及 CNN 分类模型 EfficientNetB3 的输出结果。

CNN 技术结果：
- 预测类别: {pred_class}
- 中文术语: {zh_name}
- 模型置信度: {confidence:.6f}
- 第一候选与第二候选概率差距: {margin:.6f}
- 置信度状态: {status}
- 各类别概率:
{prob_lines}

固定术语表：
- Healthy = 健康叶片
- Early_Blight = 早疫病
- Late_Blight = 晚疫病

严格规则：
1. 回答必须只使用中文。
2. 不要把结论说成绝对诊断。
3. 不要编造图像中不明显的症状。
4. CNN 是主要分类依据，VLM 负责解释和表述。
5. 不要让 VLM 替代 CNN 重新做最终分类。
6. 不要过多扩展病原学知识。
7. 不要简单把晚疫病说成“真菌病”。
8. 如果必须提到晚疫病病原，只能谨慎表述为：晚疫病与 Phytophthora infestans 相关病原有关。
9. 最后必须说明该结果属于模型辅助判断，仍建议人工复核或农业专家确认。

请严格按照以下结构输出：

诊断结论：
...

图像中的可能视觉依据：
...

说明与限制：
...
"""
    return prompt.strip()


# =========================
# 8. Qwen2.5-VL 加载与生成
# =========================
def load_qwen_vlm(model_id: str):
    print(f"\n正在加载 VLM 模型: {model_id}")
    print("第一次运行会自动下载模型文件，时间可能较长。")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28
    )

    return model, processor


def generate_with_qwen(
    image_path: Path,
    prompt: str,
    vlm_model,
    processor,
    max_new_tokens: int = 256
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path)
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(vlm_model.device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


# =========================
# 9. 输出质量检查与后处理
# =========================
def normalize_ru_terms(text: str, pred_class: str) -> Tuple[str, List[str]]:
    warnings = []
    fixed = text

    wrong_terms = {
        "ранняя плесень": CLASS_TERMS.get(pred_class, {}).get("ru", pred_class),
        "поздняя плесень": "фитофтороз",
        "поздняя гниль": "фитофтороз",
        "ранняя гниль": "альтернариоз",
        "Late_Blight (ранняя плесень)": "фитофтороз",
        "Late_Blight": "фитофтороз",
        "Early_Blight": "альтернариоз",
        "Healthy": "здоровый лист",
    }

    for wrong, correct in wrong_terms.items():
        if wrong in fixed:
            fixed = fixed.replace(wrong, correct)
            warnings.append(f"术语已修正: {wrong} -> {correct}")

    return fixed, warnings


def fallback_diagnosis(cnn_result: Dict, language: str) -> str:
    pred_class = cnn_result["predicted_class"]
    confidence = cnn_result["confidence"]
    margin = cnn_result["margin"]

    status = confidence_status(confidence, margin)

    if language == "ru":
        ru_term = CLASS_TERMS.get(pred_class, {}).get("ru_full", pred_class)

        if status == "uncertain":
            certainty = "достоверность результата ограничена"
        elif status == "cautious":
            certainty = "результат следует интерпретировать с осторожностью"
        else:
            certainty = "результат имеет высокую уверенность модели"

        return (
            "Диагностическое заключение:\n"
            f"Согласно результату CNN-классификатора, изображение относится к категории: {ru_term}. "
            f"Уверенность модели составляет {confidence * 100:.2f}%, {certainty}.\n\n"
            "Визуальные признаки на изображении:\n"
            "Изображение было использовано визуально-языковой моделью для формирования пояснения, "
            "однако конкретные признаки должны интерпретироваться осторожно и не должны заменять экспертный осмотр.\n\n"
            "Ограничения и необходимость проверки:\n"
            "Данный результат является вспомогательным выводом модели. Для окончательного заключения "
            "рекомендуется проверка агрономом или специалистом по заболеваниям растений."
        )

    zh_term = CLASS_TERMS.get(pred_class, {}).get("zh", pred_class)

    if status == "uncertain":
        certainty = "结果存在不确定性"
    elif status == "cautious":
        certainty = "结果需要谨慎参考"
    else:
        certainty = "模型置信度较高"

    return (
        "诊断结论：\n"
        f"根据 CNN 分类模型结果，该图像被判定为：{zh_term}。"
        f"模型置信度为 {confidence * 100:.2f}%，{certainty}。\n\n"
        "图像中的可能视觉依据：\n"
        "视觉语言模型已结合输入图像生成解释，但具体症状仍应结合原始图像进行人工检查。\n\n"
        "说明与限制：\n"
        "该结果属于模型辅助判断，不应作为最终农业诊断结论。建议进行人工复核或由农业专家确认。"
    )


def validate_and_postprocess(
    text: str,
    cnn_result: Dict,
    language: str
) -> Tuple[str, List[str], bool]:
    warnings = []
    need_retry = False
    processed = text

    pred_class = cnn_result["predicted_class"]

    if language == "ru":
        if contains_chinese(processed):
            warnings.append("检测到俄语输出中混入中文，需要重试。")
            need_retry = True

        processed, term_warnings = normalize_ru_terms(processed, pred_class)
        warnings.extend(term_warnings)

        if contains_chinese(processed):
            need_retry = True

    if language == "zh":
        if re.search(r"[а-яА-Я]", processed):
            warnings.append("检测到中文输出中混入俄语字符，建议检查。")

    return processed, warnings, need_retry


# =========================
# 10. 单张图完整诊断
# =========================
def diagnose_one_image(
    image_path: Path,
    cnn_model,
    cnn_transform,
    class_names: List[str],
    vlm_model,
    processor,
    language: str,
    max_new_tokens: int
) -> Dict:
    cnn_result = predict_with_cnn(image_path, cnn_model, cnn_transform, class_names)

    prompt = build_vlm_prompt(cnn_result, language=language, strict_retry=False)
    raw_text = generate_with_qwen(
        image_path=image_path,
        prompt=prompt,
        vlm_model=vlm_model,
        processor=processor,
        max_new_tokens=max_new_tokens
    )

    final_text, warnings, need_retry = validate_and_postprocess(
        raw_text,
        cnn_result,
        language
    )

    retry_used = False

    if need_retry:
        retry_used = True
        retry_prompt = build_vlm_prompt(cnn_result, language=language, strict_retry=True)
        retry_text = generate_with_qwen(
            image_path=image_path,
            prompt=retry_prompt,
            vlm_model=vlm_model,
            processor=processor,
            max_new_tokens=max_new_tokens
        )

        retry_processed, retry_warnings, retry_need = validate_and_postprocess(
            retry_text,
            cnn_result,
            language
        )

        warnings.extend(retry_warnings)

        if not retry_need:
            raw_text = retry_text
            final_text = retry_processed
        else:
            warnings.append("重试后仍存在语言或术语问题，已使用保守备用诊断文本。")
            raw_text = retry_text
            final_text = fallback_diagnosis(cnn_result, language)

    return {
        "image_path": str(image_path),
        "language": language,
        "cnn_result": cnn_result,
        "vlm_raw_output": raw_text,
        "vlm_final_output": final_text,
        "warnings": warnings,
        "retry_used": retry_used,
    }


# =========================
# 11. 保存结果
# =========================
def save_diagnosis_result(
    result: Dict,
    output_dir: Path,
    image_path: Path,
    language: str
) -> Tuple[Path, Path]:
    ensure_dir(output_dir)

    name = safe_stem(image_path)
    json_path = output_dir / f"{name}_qwen_vlm_{language}_result.json"
    txt_path = output_dir / f"{name}_qwen_vlm_{language}_result.txt"

    save_json(result, json_path)

    text_content = (
        f"Image: {result['image_path']}\n"
        f"Language: {result['language']}\n"
        f"Predicted class: {result['cnn_result']['predicted_class']}\n"
        f"Confidence: {result['cnn_result']['confidence']:.6f}\n"
        f"Margin: {result['cnn_result']['margin']:.6f}\n\n"
        f"{result['vlm_final_output']}\n"
    )

    if result["warnings"]:
        text_content += "\nWarnings:\n"
        for w in result["warnings"]:
            text_content += f"- {w}\n"

    save_text(text_content, txt_path)

    return json_path, txt_path


# =========================
# 12. 打印结果
# =========================
def print_result(result: Dict):
    cnn_result = result["cnn_result"]

    print("\n===== CNN 分类结果 =====")
    print(f"图片路径: {result['image_path']}")
    print(f"语言: {result['language']}")
    print(f"预测类别: {cnn_result['predicted_class']}")
    print(f"置信度: {cnn_result['confidence']:.6f}")
    print(f"第二候选: {cnn_result['second_class']}")
    print(f"概率差距: {cnn_result['margin']:.6f}")

    print("\n各类别概率:")
    for cls_name, prob in cnn_result["probabilities"].items():
        print(f"  {cls_name:<15}: {prob:.6f}")

    print("\n===== VLM 最终诊断结果 =====")
    print(result["vlm_final_output"])

    if result["warnings"]:
        print("\n===== 输出修正 / 警告 =====")
        for w in result["warnings"]:
            print(f"- {w}")

    if result["retry_used"]:
        print("\n提示: 本次输出使用过一次严格重试。")


# =========================
# 13. 主函数
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Pro Max: EfficientNetB3 + Qwen2.5-VL potato leaf disease diagnosis"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图片路径，支持单张图片或文件夹"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="如果输入是文件夹，是否递归读取子文件夹"
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="最多处理多少张图片，0 表示不限制"
    )

    parser.add_argument(
        "--cnn_model",
        type=str,
        default=str(DEFAULT_CNN_MODEL_PATH),
        help="EfficientNetB3 模型权重路径"
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=str(DEFAULT_CLASSES_PATH),
        help="classes.json 路径"
    )

    parser.add_argument(
        "--vlm_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Qwen2.5-VL 模型名称"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        choices=["zh", "ru", "both"],
        help="诊断输出语言：zh、ru 或 both"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="结果保存目录"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="VLM 最大生成 token 数"
    )

    args = parser.parse_args()

    input_path = Path(args.image)
    cnn_model_path = Path(args.cnn_model)
    classes_path = Path(args.classes)
    output_dir = Path(args.output_dir)

    image_paths = collect_images(
        input_path=input_path,
        recursive=args.recursive,
        max_images=args.max_images
    )

    languages = ["zh", "ru"] if args.language == "both" else [args.language]

    print("\n===== 加载 EfficientNetB3 分类模型 =====")
    class_names = load_class_names(classes_path)
    cnn_model, cnn_transform = load_cnn_model(cnn_model_path, class_names)

    print(f"使用设备: {CNN_DEVICE}")
    print(f"类别列表: {class_names}")

    print("\n===== 加载 Qwen2.5-VL 模型 =====")
    vlm_model, processor = load_qwen_vlm(args.vlm_model)

    all_results = []

    print(f"\n共找到 {len(image_paths)} 张图片。")

    for idx, image_path in enumerate(image_paths, start=1):
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(image_paths)}] 处理图片: {image_path}")

        for language in languages:
            print("\n" + "-" * 50)
            print(f"生成语言: {language}")

            try:
                result = diagnose_one_image(
                    image_path=image_path,
                    cnn_model=cnn_model,
                    cnn_transform=cnn_transform,
                    class_names=class_names,
                    vlm_model=vlm_model,
                    processor=processor,
                    language=language,
                    max_new_tokens=args.max_new_tokens
                )

                print_result(result)

                json_path, txt_path = save_diagnosis_result(
                    result=result,
                    output_dir=output_dir,
                    image_path=image_path,
                    language=language
                )

                print(f"\nJSON 已保存到: {json_path}")
                print(f"TXT 已保存到: {txt_path}")

                all_results.append(result)

            except (UnidentifiedImageError, OSError, RuntimeError, ValueError) as e:
                error_result = {
                    "image_path": str(image_path),
                    "language": language,
                    "error": str(e)
                }
                all_results.append(error_result)
                print(f"处理失败: {e}")

    summary_path = output_dir / "qwen_vlm_summary.json"
    save_json(all_results, summary_path)

    print("\n" + "=" * 70)
    print("全部处理完成。")
    print(f"总汇总文件已保存到: {summary_path}")


if __name__ == "__main__":
    main()