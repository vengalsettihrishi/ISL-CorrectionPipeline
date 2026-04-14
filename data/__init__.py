"""
data -- Preprocessing pipeline for iSign ISL recognition.

Modules:
    landmark_loader  : Load / extract MediaPipe landmarks
    velocity         : Compute frame-to-frame velocity features
    feature_builder  : Concatenate, normalize, pad/truncate features
    label_encoder    : Tokenize English translations, build vocabulary
    dataset          : PyTorch Dataset + DataLoader + collate_fn
    augmentation     : Data augmentation transforms for landmarks
"""
