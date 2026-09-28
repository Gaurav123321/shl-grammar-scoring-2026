# Grammar Scoring Engine — SHL Hiring Assessment 2026

This repository contains the Kaggle submission notebook for predicting continuous grammar-MOS scores (0–5) from spoken English WAV recordings.

## Approach

The pipeline combines:

- Faster-Whisper English transcripts (`small.en`)
- Word and character TF-IDF features from the transcript
- Acoustic features: duration, energy, silence ratio, zero-crossing rate, spectral centroid, and MFCC summaries
- Ridge regression with 5-fold out-of-fold validation

## Validation result

The notebook achieved:

- 5-fold OOF RMSE: **0.8025**
- 5-fold OOF Pearson correlation: **0.7618**

The initial Kaggle public-leaderboard submission scored **0.7291**.

## Run on Kaggle

1. Create a Kaggle notebook and attach the competition dataset.
2. Upload or import `grammar_scoring_solution.ipynb`.
3. Enable Internet and a GPU T4 accelerator.
4. Run all cells. The notebook writes `submission.csv`.

The first run downloads the ASR model and transcribes the recordings. Transcripts are cached in `cache/transcripts.csv` during the session.

## Requirements

```bash
pip install faster-whisper librosa soundfile scikit-learn pandas seaborn
```

The competition audio and CSV data are not included in this repository.
