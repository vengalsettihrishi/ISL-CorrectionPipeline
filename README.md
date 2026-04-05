# ISL Word-Level Recognition Pipeline — Stage 1

## Project Structure

```
isl_pipeline/
├── config.py              # All hyperparameters and paths in one place
├── landmark_extractor.py  # MediaPipe Hands + Pose extraction from video
├── dataset.py             # PyTorch Dataset for landmark sequences
├── model.py               # Bidirectional LSTM classifier
├── train.py               # Training loop with validation
├── evaluate.py            # Evaluation metrics (F1, confusion matrix)
└── README.md
```

## How to use

### Step 1: Extract landmarks from video dataset
```bash
python landmark_extractor.py --input_dir /path/to/INCLUDE50/videos --output_dir /path/to/processed
```

### Step 2: Train the model
```bash
python train.py --data_dir /path/to/processed --epochs 50
```

### Step 3: Evaluate
```bash
python evaluate.py --data_dir /path/to/processed --checkpoint best_model.pth
```

## Dataset Format Expected

The INCLUDE-50 dataset should be organized as:
```
videos/
├── Hello/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── ThankYou/
│   ├── video_001.mp4
│   └── ...
└── ...
```

Each subfolder name = class label.
Each file = one isolated sign video (2-3 seconds).
