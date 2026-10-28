# MediAI: نظام مساعد لتحليل أشعة الصدر (CDSS - Prototype)


A model that analyzes chest X-ray images and classifies them as Normal or Pneumonia, then displays the Grad-CAM heatmap and a preliminary report based on specific medical sources.
Disclaimer: This system is not a medical diagnostic tool. The results are preliminary and require review by a specialist physician.

---

## 1.  system boundaries

-Pneumonia Detection Only: Detects Pneumonia versus Normal chest X-rays. It does not detect tumors, nodules, or other thoracic pathologies.
-No Etiologic Sub-typing: Does not differentiate between bacterial and viral pneumonia, even though the dataset contains both types.
-Single-Source Dataset: Trained and evaluated on a single dataset; out-of-distribution performance on images from other clinical sources remains unverified.


---

## 2. dataset

### source

- **name:** Chest X-Ray: bacterial / viral pneumonia, normal
- **link:** (https://www.kaggle.com/datasets/kostasdiamantaras/chest-xrays-bacterial-viral-pneumonia-normal)
[kaggle.com/datasets/kostasdiamantaras/chest-xrays-bacterial-viral-pneumonia-normal]

### (Labels)
| class_id | number of images in training |
|---|---|
| 0 | Normal | 1227 |
| 1 | Bacterial pneumonia | 2238 |
| 2 | Viral pneumonia | 1207 |


### binary conversion

binary classification:
result:1227 normal & 3445 pneumonia & using pos_weight in loss function.


### structure data

- `train_images/`: 4672 images ،  **labels**  `labels_train.csv`.
- `test_images/`: 1168 images ، **No labels**.

| Train | 70% | ~3270 |
| Validation | 15% | ~701 |
| Test | 15% | 701 (NORMAL: 184، PNEUMONIA: 517) |


---

## 3.Data Extraction

```bash
unzip -q chest-xrays-bacterial-viral-pneumonia-normal.zip -d /content/data

/content/data/
├── labels_train.csv          (file_name, class_id)
├── train_images/
│   └── train_images/         (4672 images)
└── test_images/
    └── test_images/          (1168 images)




Backbone: EfficientNet-B0 ptrtrain by ImageNet (EfficientNet_B0_Weights.DEFAULT).
Head :
Dropout(p=0.3)
Linear(1280 → 1)   ← logit 
Output: logit، 
loss function 
BCEWithLogitsLoss مع pos_weight = n_normal / n_pneumonia 

Optimizer: AdamW، weight_decay = 1e-4.
Scheduler: CosineAnnealingLR.
Batch size: 32.
Seed: 42.

data augmentation on train only.
RandomHorizontalFlip
RandomRotation (±10°)
RandomAffine (translate 5%، scale 0.95 لـ 1.05)
ColorJitter (brightness 0.2، contrast 0.2)

-Validation & Testing Pipeline: The Validation and Test sets only apply standard image transformations (e.g., resizing, normalization) without data augmentation to maintain objective evaluation.
-Checkpointing Strategy: Model checkpoints are saved based on the highest Validation AUC across all epochs, rather than taking the final epoch.
-Optimal Decision Thresholding:
 --To handle class imbalance, the default decision threshold (0.5) is overridden.
 --The optimal threshold is selected from the validation set by maximizing the F1-score along the Precision-Recall Curve.
 --The selected threshold value is dynamically saved to models/threshold.json and consumed directly by the Streamlit web application   during inference.



## result:
Precision
Recall
F1
Support
NORMAL
0.97
0.91
0.94
184
PNEUMONIA
0.97
0.99
0.98
517
Accuracy


0.97
701
Confusion Matrix: [[167, 17], [5, 512]]
ROC-AUC: 0.9954



## project Structure
├── app.py                    
├── models/
│   ├── efficientnet_pneumonia.pth  
│   ├── threshold.json               
├── knowledge/
│   ├── normal.md         
│   ├── pneumonia.md           
│   └── general.md         
├── src/
│   ├── model.py         
│   ├── rag.py           
│   └── llm_helper.py         
└── .env                      



# 🫁 Medical Image Diagnosis System & RAG Assistant

A Clinical Decision Support System (CDSS) for Chest X-Ray diagnostic support (Pneumonia Detection) built using Deep Learning and Retrieval-Augmented Generation (RAG).

---

## 📌 RAG-based Medical Knowledge Integration

* **Retrieval Method:** TF-IDF-based document retrieval that fetches relevant medical context based on the classification result (`NORMAL` or `PNEUMONIA`) alongside patient-reported symptoms.
* **Knowledge Sources:** Handcrafted medical context compiled from trusted public sources including **WHO**, **CDC**.
* **Medical Disclaimer:** Content is for academic and decision-support purposes and requires formal clinical review before any real-world medical deployment.

---

## 📄 Automated Report Generation

* **LLM Engine:** Powered by **Groq API** for high-speed inference.
* **Prompt Guardrails:** 
  * Strictly prohibits final medical diagnoses.
  * Prevents specific drug prescriptions, dosages, or etiologic sub-typing (bacterial vs. viral).
* **Mandatory Disclosure:** Every report concludes with the required disclaimer:
  > *"X-rays do not rule out conditions other than pneumonia. A comprehensive medical evaluation by a licensed physician is required."*

---

## 🚀 Getting Started & Execution

### 1. Prerequisites & Installation

Install the required dependencies using `pip`:

```bash
pip install torch torchvision streamlit pillow opencv-python-headless scikit-learn pandas groq python-dotenv


# demo 
# by Mohamed said abdalaziz
