# Parkinson Speech Progression Models

Implementation scaffold adapted from the paper:

Detection and Forecasting of Parkinson Disease Progression from Speech Signal Features Using MultiLayer Perceptron and LSTM, arXiv:2412.18248.

The paper used the UCI Parkinson's Telemonitoring Dataset, where features and longitudinal progression labels are already tabular. This workspace contains the Figshare dataset "Voice Samples for Patients with Parkinson's Disease and Healthy Controls", which contains WAV files in folders such as `HC_AH` and `PD_AH`.

So the project trains:

- `mlp`: Parkinson detection from selected speech features.
- `sequence`: only when you have repeated observations per subject with a real ordered variable. This is not the default task for the Figshare dataset.

## Figshare WAV Dataset

First extract features from WAV files:

```bash
python scripts/extract_figshare_features.py --root . --out data/figshare_features.csv
```

Then train PD vs healthy-control detection:

```bash
python train.py \
  --task detection \
  --csv data/figshare_features.csv \
  --target-col pd_label \
  --feature-selection relief \
  --num-features 20 \
  --epochs 100
```

Or use the YAML config:

```bash
python train.py --config configs/figshare_detection.yaml
```

Training writes:

- `runs/detection_mlp.pt`: checkpoint containing model weights, merged args, selected feature columns, and class names.
- `runs/detection_mlp_metrics.json`: final metrics, selected features, confusion matrix, and epoch history.
- `runs/detection_mlp_train.log`: per-epoch CSV log with `epoch,train_loss,val_accuracy,val_f1`.

The generated CSV includes `pd_label`, `group`, `subject_id`, `utterance`, and acoustic features such as RMS, ZCR, spectral features, MFCC summaries, pitch summaries, entropy, and jitter/shimmer proxies.

## Generic Tabular Data Format

Use a CSV where each row is one speech-feature observation.

Required columns for detection:

- Feature columns: numeric acoustic/speech features.
- Detection label column, default: `pd_label`, with values `0/1`.

Required columns for progression forecasting:

- Patient id column, default: `subject_id`.
- Time/order column, default: `visit`.
- Stage label column, default: `stage`, e.g. `2` and `3`.
- Numeric feature columns.

## UCI Parkinson's Telemonitoring Dataset

The UCI Telemonitoring dataset is a regression dataset containing only Parkinson's patients. It is used to predict `motor_UPDRS` or `total_UPDRS`, not healthy-vs-PD detection.

Run tabular MLP regression:

```bash
python train.py --config configs/tele_mlp_motor.yaml
```

Run sequence regression by patient timeline:

```bash
python train.py --config configs/tele_lstm_motor.yaml
python train.py --config configs/tele_spmamba_motor.yaml
```

Regression logs contain:

- `train_loss`: scaled target MSE used for optimization.
- `val_loss`: scaled target MSE on validation.
- `val_rmse`: RMSE in the original UPDRS units.
- `val_mae`: MAE in the original UPDRS units.
- `val_r2`: coefficient of determination.

Example:

```csv
subject_id,visit,jitter,shimmer,hnr,mfcc_1,mfcc_2,pd_label,stage
p001,1,0.01,0.04,20.1,0.5,-0.2,1,2
p001,2,0.02,0.05,18.8,0.4,-0.1,1,3
p002,1,0.00,0.02,25.0,0.1,-0.3,0,0
```

## Install

```bash
pip install -r requirements.txt
```

`mamba-ssm` is optional. If it is unavailable, `spmamba` uses a lightweight PyTorch fallback with Mamba-like gated state mixing so the pipeline still runs.

## Make Sample Data For Pipeline Debugging

The commands below use `data/features.csv`. Create a synthetic CSV first if you do not have the real speech-feature dataset yet:

```bash
python scripts/make_sample_data.py --out data/features.csv
```

## Train MLP Detection

```bash
python train.py \
  --task detection \
  --csv data/features.csv \
  --target-col pd_label \
  --feature-selection relief \
  --num-features 20 \
  --epochs 100
```

## Train Progression Forecasting with LSTM

```bash
python train.py \
  --task progression \
  --csv data/features.csv \
  --subject-col subject_id \
  --time-col visit \
  --target-col stage \
  --sequence-backbone lstm \
  --epochs 100
```

Use this only if your CSV has true longitudinal columns such as `subject_id`, `visit`, and a progression target such as `stage` or UPDRS-derived bins.

## Train Progression Forecasting with SpMamba

```bash
python train.py \
  --task progression \
  --csv data/features.csv \
  --subject-col subject_id \
  --time-col visit \
  --target-col stage \
  --sequence-backbone spmamba \
  --epochs 100
```

## Notes

The paper uses Relief-F and Sequential Forward Selection before MLP/LSTM. This implementation supports:

- `--feature-selection none`
- `--feature-selection relief`
- `--feature-selection sfs`

For a fair LSTM vs SpMamba comparison, keep all preprocessing, selected features, splits, optimizer settings, and epochs identical.
