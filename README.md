# Potato Leaf Disease Recognition with CNN and VLM

This project implements a potato leaf disease recognition workflow based on deep learning and vision-language diagnosis generation.

The main task is to classify potato leaf images into three categories:

- Early Blight
- Healthy
- Late Blight

The project also connects the CNN classification result with Qwen2.5-VL to generate a natural-language diagnostic description.

## Project Overview

The workflow is:

```text
Input potato leaf image
        ↓
CNN disease classification
        ↓
Predicted class and confidence
        ↓
Qwen2.5-VL diagnostic explanation
        ↓
Natural-language diagnosis output
```

The main classification model is EfficientNetB3. ResNet50 is used as a baseline model for comparison.

## Models

This project includes the following experiments:

| Model | Dataset | Task |
|---|---|---|
| ResNet50 | Pak-PLD | Baseline classification |
| EfficientNetB3 | Pak-PLD | Main classification model |
| EfficientNetB3 | PlantVillage Potato | Controlled dataset validation |
| Qwen2.5-VL | Sample images | Diagnostic text generation |

## Datasets

Two potato leaf datasets were used:

| Dataset | Classes |
|---|---|
| Pak-PLD Potato | Early_Blight, Healthy, Late_Blight |
| PlantVillage Potato | Early_Blight, Healthy, Late_Blight |

Both datasets are included in this repository for reproducibility.

## Experimental Results

| Model | Dataset | Accuracy | Macro F1-score |
|---|---|---:|---:|
| ResNet50 | Pak-PLD | 0.9901 | 0.9891 |
| EfficientNetB3 | Pak-PLD | 0.9975 | 0.9972 |
| EfficientNetB3 | PlantVillage Potato | 1.0000 | 1.0000 |

EfficientNetB3 achieved the best performance and was selected as the main classification model.

## Main Files

```text
main.py                                  # CNN inference script
main_qwen_vlm.py                         # CNN + Qwen2.5-VL diagnosis script
train_resnet50_pakpld.py                 # ResNet50 training on Pak-PLD
train_efficientnetb3_pakpld.py           # EfficientNetB3 training on Pak-PLD
train_efficientnetb3_plantvillage.py     # EfficientNetB3 training on PlantVillage
```

## Results Folders

```text
runs_resnet50_pakpld/
runs_efficientnetb3_pakpld/
runs_efficientnetb3_plantvillage/
demo_outputs_vlm/
```

These folders contain trained model weights, training curves, confusion matrices, classification reports, summaries, and prediction outputs.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Example Usage

Run CNN inference:

```bash
python main.py --image sample_images/Early_Blight_1.jpg
```

Run CNN + VLM diagnosis:

```bash
python main_qwen_vlm.py --image sample_images --language ru
```

## Notes

- Trained model weight files are included in the repository.
- The Pak-PLD-Potato and PlantVillage-Potato datasets are included for reproducibility.
- This repository contains code, experiment results, figures, and thesis-related materials.
- YOLO-based leaf detection is not included in the current implementation and is reserved for future work.
- Qwen2.5-VL may require internet access for the first model download and sufficient hardware resources.
- The default CNN inference script uses:
  `runs_efficientnetb3_pakpld/models/best_efficientnetb3_pakpld.pth`

## Future Work

Future improvements may include:

- Adding YOLO-based potato leaf detection for real field images
- Testing with more real-world field images
- Improving multilingual diagnosis generation
- Deploying the system as a web or mobile application
