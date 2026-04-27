# Gloss-Guided Structural Regularization

This folder implements the two-phase experiment:

1. Train a gloss CTC model on a small gloss dataset.
2. Freeze the gloss head and use its blank/non-blank timing signal to train
   English CTC on iSign.

The important rule is: iSign is never trained with gloss labels. Gloss labels
only teach the model temporal sign structure.

## 1. Prepare Pose Files

Both datasets must have pose files with the same naming rule:

```text
poses/
  sample_001.npy
  sample_002.npy
```

Each `.npy` file may be either:

```text
(T, 225) raw landmarks
```

or:

```text
(T, 450) [landmarks, velocity]
```

If the file is `(T, 225)`, the code automatically adds velocity to make
`(T, 450)`.

## 2. Prepare CSV Manifests

Gloss dataset:

```csv
video_id,gloss
sample_001,I EAT FOOD
sample_002,YOU GO SCHOOL
```

iSign English dataset:

```csv
video_id,english
2011b172755b-67,"Come to me, Chicks. I want you here."
```

Your existing file already matches this:

```text
data_iSign/iSign_v1.1.csv
data_iSign/poses/
```

## 3. Phase 1: Train Gloss Structure Model

Replace the gloss paths with your small gloss dataset paths:

```powershell
python -m gloss_structure.train_phase1_gloss `
  --manifest data_gloss/gloss.csv `
  --pose_dir data_gloss/poses `
  --label_column gloss `
  --out_dir checkpoints/gloss_structure_phase1 `
  --epochs 80 `
  --batch_size 8 `
  --lr 1e-4 `
  --dropout 0.3
```

This saves:

```text
checkpoints/gloss_structure_phase1/best_gloss_model.pth
checkpoints/gloss_structure_phase1/gloss_vocab.json
checkpoints/gloss_structure_phase1/norm_stats.npz
```

## 4. Plot The Structural Signal

Use one gloss pose file:

```powershell
python -m gloss_structure.plot_structure `
  --checkpoint checkpoints/gloss_structure_phase1/best_gloss_model.pth `
  --norm_stats checkpoints/gloss_structure_phase1/norm_stats.npz `
  --pose_file data_gloss/poses/sample_001.npy `
  --out_png results/sample_001_qt.png
```

The plotted value is:

```text
q_t = 1 - P(gloss_blank at frame t)
```

High values should roughly align with active signing. If the plot is random,
the gloss model has not learned useful structure yet.

## 5. Phase 2: Train iSign English With Structural Loss

```powershell
python -m gloss_structure.train_phase2_isign `
  --manifest data_iSign/iSign_v1.1.csv `
  --pose_dir data_iSign/poses `
  --label_column english `
  --gloss_checkpoint checkpoints/gloss_structure_phase1/best_gloss_model.pth `
  --gloss_norm_stats checkpoints/gloss_structure_phase1/norm_stats.npz `
  --out_dir checkpoints/gloss_structure_phase2 `
  --english_token_mode char `
  --lambda_struct 0.05 `
  --head_epochs 5 `
  --full_epochs 60 `
  --batch_size 16 `
  --lr 1e-4
```

This does two stages:

```text
Stage A: freeze encoder + gloss head, train English head only
Stage B: unfreeze encoder, keep gloss head frozen, train with structural loss
```

The loss is:

```text
L = English_CTC + lambda_struct * BCE(r_t, stopgrad(q_t))
```

where:

```text
q_t = 1 - P(gloss_blank)
r_t = 1 - P(english_blank)
```

## 6. Run The Key Ablation

Pretraining without structural loss:

```powershell
python -m gloss_structure.train_phase2_isign `
  --manifest data_iSign/iSign_v1.1.csv `
  --pose_dir data_iSign/poses `
  --label_column english `
  --gloss_checkpoint checkpoints/gloss_structure_phase1/best_gloss_model.pth `
  --gloss_norm_stats checkpoints/gloss_structure_phase1/norm_stats.npz `
  --out_dir checkpoints/gloss_structure_no_struct `
  --lambda_struct 0.0
```

Then compare validation token error in each `history.json`.

Try these structural weights:

```text
0.01, 0.05, 0.1
```

Start with `0.05`.

