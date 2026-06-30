# Plant Disease Detection System (CNN From Scratch)

This is a complete AI/ML mini-project for a 4th semester group:
- image classification of plant leaf diseases,
- custom CNN architecture built from scratch,
- full training and testing pipeline,
- inference for a single image.

## 1) Project objective

Build an automated system that predicts plant disease class from a leaf image.

## 2) "From scratch" scope

This project uses:
- a **custom-designed CNN architecture** (not pre-trained models),
- training from random initialization,
- your own train/val/test split,
- your own evaluation report.

This project does **not** use transfer learning (ResNet, EfficientNet, etc.).

## 3) Dataset format expected
The dataset used for this project is:

**New Plant Diseases Dataset (Augmented)**  
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Download the dataset from Kaggle and place it inside the `data/` directory before training the model.

Put your raw dataset in:

`data/raw/`

Structure:

data/raw/
  Tomato___Healthy/
    img1.jpg
    img2.jpg
  Tomato___Late_blight/
    ...
  Potato___Early_blight/
    ...

Each folder name is the class label.

## 4) Setup

```bash
pip install -r requirements.txt
```

## 5) Run steps

### Step A: Split dataset
```bash
python scripts/split_data.py --source data/raw --dest data/processed --train 0.7 --val 0.15 --test 0.15
```

### Step B: Train model
```bash
python train.py --data_dir data/processed --epochs 20 --batch_size 32 --lr 0.001
```

### Step C: Evaluate model
```bash
python evaluate.py --data_dir data/processed --checkpoint models/best_model.pth
```

### Step D: Predict for one image
```bash
python predict.py --image path/to/leaf.jpg --checkpoint models/best_model.pth
```

### Step E: Run Streamlit UI
```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## 6) Output files

- Best model: `models/best_model.pth`
- Last model: `models/last_model.pth`
- Training curves: `reports/training_curves.png`
- Classification report: `reports/classification_report.txt`
- Confusion matrix: `reports/confusion_matrix.png`
