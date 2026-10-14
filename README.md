# Conformal-Aware Selective Abstention of Zero-Shot Vision-Language Models under Semantic and Visual Distribution Shift

## 1. Project Overview

This repository implements an evaluation pipeline for post-hoc calibration, conformal prediction set construction, selective abstention, and out-of-distribution (OOD) detection using a frozen, zero-shot OpenAI CLIP `ViT-B/32` vision-language model (`openai-clip/ViT-B/32`). All predictions are made over the 10-class CIFAR-10 label vocabulary across four natural-language prompt templates (`P0`–`P3`) and five deterministic random seeds (`[13, 42, 123, 456, 765]`).

The pipeline addresses a core operational requirement in zero-shot vision-language deployment: determining when a frozen classifier should abstain from answering or flag an input as out-of-distribution under semantic shift, natural phrasing shift, and visual corruption, without fitting parameters on test or OOD data. All post-hoc calibration parameters—a scalar softmax temperature $T > 0$ and a finite-sample split-conformal nonconformity quantile $\hat{q}_{1-\alpha}$ at miscoverage level $\alpha = 0.10$—as well as all selective-prediction thresholds (for fixed nominal coverage targets and maximum coverage under a risk bound) are fitted exclusively on seeded 5,000-sample subsets drawn from the 50,000-sample CIFAR-10 training split.

During execution, the code:
1. Downloads or reuses persistent copies of CIFAR-10 (train and test), CIFAR-10-C (six corruption families at severities 1, 3, and 5), CIFAR-100 (test), SVHN (test), and OpenAI CLIP `ViT-B/32` weights.
2. Encodes and caches L2-normalized text embeddings (`[10, 512]`) for four prompt templates and L2-normalized image embeddings (`[N, 512]`) and 10-class logits (`[N, 10]`) for the calibration bank and 21 held-out dataset conditions using SHA-256 configuration fingerprints.
3. Fits per-seed temperature scaling parameters and split-conformal quantiles on the CIFAR-10 training calibration subsets and caches calibration-side confidence and correctness vectors for six uncertainty methods (`MSP`, `entropy`, `margin`, `temperature_MSP`, `prompt_agreement`, and `conformal_set_size`).
4. Evaluates all six methods across 10 experiment families (`E1`–`E10`, expanded to 30 dataset/corruption/severity/prompt conditions) and 5 seeds, computing fixed-coverage selective risk, risk-constrained coverage, full-curve AURC/EAURC, calibration metrics (ECE, NLL, multiclass Brier score), and OOD detection metrics (AUROC, AUPR-Out, FPR95).
5. Computes 2,000-resample percentile bootstrap 95% confidence intervals for primary conditions, extracts diagnostic failure cases, generates summary CSV/Markdown tables and PNG/PDF figures, runs a 34-check automated validation audit, and packages all generated files into downloadable archives.

## 2. Project Structure

```text
.
├── README.md
├── report.tex
├── report.pdf
├── code_changes.py
├── conformal_aware_selective_abstention_clip_colab_persistent_drive.ipynb
└── conformal_aware_selective_abstention_clip_colab_persistent_drive.py
```

### Role of Supplied and Delivered Files

- `conformal_aware_selective_abstention_clip_colab_persistent_drive.ipynb`: Primary 59-cell Jupyter notebook (27 Markdown cells and 32 code cells) implementing the complete 26-stage pipeline with saved execution outputs from a Google Colab Tesla T4 runtime. (Reviewed from the supplied notebook file `conformal_aware_selective_abstention_clip_colab_persistent_drive (3).ipynb`.)
- `conformal_aware_selective_abstention_clip_colab_persistent_drive.py`: Synchronized standalone Python script representation of the notebook. All 32 code cells and 27 Markdown section headers are preserved in exact execution order, with IPython `%pip` magic commands commented out for standard Python syntax compatibility.
- `code_changes.py`: Python audit module recording all substantive cleanups, consolidations of duplicate definitions, warning fixes, reviewed file summaries, and retained implementation behaviors (`CHANGES`, `REVIEW_SUMMARY`, and `UNRESOLVED_ITEMS`).
- `README.md`: Project documentation covering architecture, workflow, dependencies, setup, execution, input/output schemas, and implementation constraints.
- `report.tex` and `report.pdf`: Comprehensive LaTeX technical report and compiled PDF detailing the software architecture, mathematical definitions, algorithms, configuration handling, and validation procedures.

### Runtime Directory Structure Created During Execution

When executed, the pipeline creates and populates a persistent directory tree under `PERSISTENT_PROJECT_ROOT` (`/content/drive/MyDrive/conformal_vlm_research` when Google Drive is mounted in Colab, or `/content/conformal_vlm_research` otherwise):

```text
conformal_vlm_research/
├── data/
│   ├── cifar10/                  # Torchvision CIFAR-10 train (50,000) and test (10,000)
│   ├── cifar100/                 # Torchvision CIFAR-100 test (10,000)
│   ├── svhn/                     # Torchvision SVHN test (26,032)
│   ├── CIFAR-10-C.tar            # Downloaded Zenodo archive (if not already extracted)
│   └── CIFAR-10-C/               # Extracted .npy corruption arrays and labels.npy
├── cache/
│   ├── clip/                     # Downloaded OpenAI CLIP ViT-B/32 checkpoint
│   ├── text_embeddings/          # Fingerprinted .npz/.json caches for P0-P3 text embeddings
│   ├── image_embeddings/         # Fingerprinted .npz/.json caches for image embeddings and labels
│   └── logits/                   # Fingerprinted .npz/.json caches for per-prompt logits
├── output/
│   ├── all_results.csv           # Condition-level selective, calibration, curve, and OOD summaries
│   └── per_sample_records.csv    # Sample-level predictions, confidences, uncertainties, and set sizes
├── tables/                       # 11 CSV summary tables and 8 Markdown table exports
├── figures/                      # 13 PNG (200 DPI) and 13 vector PDF visualizations
├── metadata/                     # Split indices, seed calibration parameters, provenance, and audit CSV
└── final_output/                 # Consolidated export directory packaged into ZIP archives
```

External datasets (`CIFAR-10`, `CIFAR-100`, `SVHN`, `CIFAR-10-C`) and the CLIP `ViT-B/32` model checkpoint are not bundled inside the repository archive; the code checks `conformal_vlm_research/data/` and `conformal_vlm_research/cache/clip/` at startup and downloads missing assets automatically when outbound network access is available.

## 3. Implementation Overview

The pipeline executes sequentially across 26 logical sections:

1. **Environment and Reproducibility (Sections 1–2, Cells 01–06)**: Imports core libraries, mounts Google Drive if running in Colab, defines `CONFIG`, `PROMPTS`, `CORRUPTIONS`, `SEVERITIES`, and the 10-row `EXPERIMENTS` table (`E1`–`E10`), creates directory roots (`ROOTS`), and initializes Python, NumPy, and PyTorch random seeds via `set_global_seed(13)` and `seed_numpy`.
2. **Validation, Fingerprinting, and Caching Utilities (Section 3, Cells 07–08)**: Defines `canonical_json`, `sha256_text`, and `config_fingerprint` (which hashes `cache_version`, `model_identifier`, `PROMPTS`, `CIFAR10_CLASSES`, and stage-specific metadata into a 16-character hex digest). Defines `assert_finite`, `validate_probabilities`, atomic `.npz`/`.json` cache I/O (`save_array_cache`, `load_array_cache`), and JSON helpers (`save_json`, `load_or_none_json`).
3. **Dataset Acquisition and Integrity Verification (Section 4, Cells 09–12)**: Checks for required files under `DATA_ROOT` via `_dataset_present` before loading `CIFAR10` (train and test), `CIFAR100` (test), and `SVHN` (test). Downloads `CIFAR-10-C.tar` via `download_with_progress` if needed, validates tar headers with `validate_tar`, extracts the six configured `.npy` corruption files and `labels.npy` via `extract_cifar10c_tar`, and verifies that each severity slice (`10,000` images of shape `(32, 32, 3)`) matches `cifar10_test_raw.targets`.
4. **Calibration Splits and Leakage Guardrails (Section 5, Cells 13–14)**: `make_calibration_split(seed, n_total=50000, n_cal=5000)` samples 5,000 unique sorted indices from the CIFAR-10 training split for each seed in `CONFIG["seeds"]`, storing them in `CAL_SPLITS`. Offsets test indices by `TRAIN_TOTAL = 50000` (`TEST_INDICES` in `[50000, 59999]`), asserts zero overlap between `CAL_SPLITS` and `TEST_INDICES`, and saves `split_metadata.json`.
5. **Model Loading, Prompt Formatting, and Dataset Adapters (Sections 6–8, Cells 15–21)**: Loads `ViT-B/32` via `clip.load(..., jit=False)`, freezes all parameters (`requires_grad_(False)`), formats the 10 class names across `P0`–`P3` (`build_prompts`, `build_all_prompt_texts`), and caches L2-normalized text embeddings (`get_text_embeddings` in `TEXT_EMBEDDINGS`). Defines `ArrayImageDataset` and `TorchvisionImageDataset` to convert images to RGB `PIL.Image` instances, apply CLIP `preprocess`, and map SVHN label `10` to `0`.
6. **Batched Image Encoding and Logit Bank Construction (Sections 9–10, Cells 22–25)**: `encode_image_dataset` encodes images in batches (using `torch.autocast` in FP16 on CUDA when `use_amp=True`), L2-normalizes features, and caches them. `compute_prompt_logits` multiplies image and text embeddings by `model.logit_scale.exp()`. `get_condition_logits` and `_validate_prompt_logits_bank` verify and cache `logits_P0`–`logits_P3`. Cell 25 sets `CONFIG["num_workers"] = 0` for Colab stability and builds `CALIBRATION_BANK` (`CIFAR-10` train, 50,000 samples) and `DATA_BANK` (21 held-out test entries: `E1`, `E7`, `E8`, and 18 `C10C::<corruption>::s<severity>` entries).
7. **Post-Hoc Calibration and Uncertainty Extraction (Sections 11–13, Cells 26–31)**:
   - `softmax_np`, `entropy_np`, `top12_stats`, `prediction_from_probs`, `confidence_uncertainties`, and `temperature_scale_logits` compute row-normalized probabilities and base confidence/uncertainty scores.
   - `fit_temperature` optimizes scalar temperature $T \in [0.05, 10.0]$ via `scipy.optimize.minimize_scalar(..., method="bounded")` to minimize multiclass NLL on the seed's `P0` calibration logits.
   - `fit_seed_calibration` computes split-conformal nonconformity scores $s_i = 1 - p_{\text{base}}(y_i \mid x_i)$ on the $n = 5000$ calibration samples and selects the empirical quantile at level $\min(1, \lceil (n + 1)(1 - \alpha) \rceil / n) = 4501 / 5000 = 0.9002$ using `method="higher"`.
   - `prompt_level_outputs` computes vectorized per-prompt predictions, majority-vote class, prompt agreement, and prompt instability across `P0`–`P3`. `conformal_sets` forms prediction sets $C(x) = \{y : p_{\text{base}}(y \mid x) \ge 1 - \hat{q}_{1-\alpha} - 10^{-12}\}$ with a top-1 fallback if empty.
8. **Selective Prediction, Evaluation, and Statistical Analysis (Sections 14–20, Cells 32–46)**:
   - `threshold_for_target_coverage`, `selective_metrics`, `max_coverage_under_risk`, `risk_coverage_curve`, `aurc`, `oracle_aurc`, `eaurc`, `fpr95`, `ood_metrics`, `ece_score`, `nll_score`, and `brier_score_multiclass` implement all evaluation metrics.
   - Cell 39 precomputes calibration-side confidence and correctness vectors in `CAL_FEATURES[seed]` for all six methods (`MSP`, `entropy`, `margin`, `temperature_MSP`, `prompt_agreement`, `conformal_set_size`).
   - Cell 41 expands `EXPERIMENTS` into the 30-row `FULL_MATRIX`, and Cell 42 evaluates all 30 conditions across 5 seeds and 6 methods via `evaluate_seed_condition` and `ood_metrics`, writing `all_results.csv` (6,360 rows) and `per_sample_records.csv` (9,961,920 rows).
   - Cell 44 aggregates summary tables by operating rule, and Cell 46 computes 2,000-resample bootstrap 95% confidence intervals (`bootstrap_risk_ci_progress`, `bootstrap_aurc_ci_progress`, `bootstrap_auroc_ci_progress`), writing `bootstrap_95ci.csv` (360 rows).
9. **Visualization, Failure Analysis, Validation, and Export (Sections 21–26, Cells 47–58)**: Generates 13 PNG and 13 PDF figures (Cell 48), extracts top-25 diagnostic failure cases per category (Cell 50), exports CSV and Markdown comparison tables (Cell 52) and provenance JSON (Cell 54), runs the 34-check `final_validation()` audit (Cell 56), and packages all outputs into `final_output/` and downloadable ZIP archives (Cell 58).

## 4. Requirements and Environment

### Python and Package Dependencies

The implementation imports the following packages:
- **Python standard library**: `os`, `sys`, `json`, `math`, `hashlib`, `random`, `shutil`, `tarfile`, `zipfile`, `platform`, `warnings`, `urllib.request`, `pathlib`, `datetime`
- **Core scientific and data libraries**: `numpy`, `pandas`, `scipy`, `scikit-learn` (`sklearn`), `matplotlib`, `Pillow` (`PIL`), `tqdm`
- **Deep learning and vision-language libraries**: `torch`, `torchvision`, `clip` (installed from `git+https://github.com/openai/CLIP.git`) and its text-tokenization dependencies `ftfy` and `regex`
- **Optional environment integration**: `google.colab` (automatically detected via `try ... except ImportError`; used only for Drive mounting and browser file downloads) and `tabulate` (required by `pandas.DataFrame.to_markdown()` in Section 23)

### Reference Runtime vs. General Compatibility

No `requirements.txt` or `pyproject.toml` file was supplied with the project. The code enforces `sys.version_info >= (3, 9)` in `final_validation()` and uses `np.trapezoid` (introduced in NumPy 2.0) and `np.quantile(..., method=...)` (introduced in NumPy 1.22).

The saved notebook outputs record the exact Google Colab environment in which the full pipeline was executed:
- Python `3.13.15`
- PyTorch `2.11.0+cu128` and torchvision `0.26.0+cu128` (CUDA `12.8`, NVIDIA Tesla T4 GPU with 14.56 GiB VRAM)
- NumPy `2.1.3`, pandas `2.2.3`, scikit-learn `1.6.1`, SciPy `1.16.3`

### Hardware Assumptions

CUDA is optional but recommended for initial image encoding (`encode_image_dataset`). The code automatically falls back to CPU (`DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`), reduces `batch_size` from `128` to `32`, and disables automatic mixed precision (`use_amp = False`) when CUDA is unavailable.

## 5. Installation and Setup

To set up a local or cloud Python environment (Python 3.10+ with NumPy 2.0+ required for `np.trapezoid` and `int | None` type-union syntax):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib pillow tqdm tabulate ftfy regex
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
```

In Google Colab, PyTorch and torchvision are pre-installed, and Cell 02 of the notebook installs the remaining packages directly:

```python
%pip install -q ftfy regex tqdm scikit-learn scipy pandas matplotlib
%pip install -q git+https://github.com/openai/CLIP.git
```

## 6. How to Run the Project

### Option A: Running the Jupyter Notebook (Recommended)

Open `conformal_aware_selective_abstention_clip_colab_persistent_drive.ipynb` in Google Colab or a local Jupyter server and execute all cells in order from top to bottom (**Runtime -> Run all**).

- **Cell Execution Order**: Strict top-to-bottom execution is required. Later cells explicitly check `globals()` for upstream structures (`CAL_SPLITS`, `CALIBRATION_BANK`, `DATA_BANK`, `SEED_CALIBRATIONS`, `CAL_FEATURES`, `FULL_MATRIX`, `summary_df`, `sample_df`, and `validation_df`) and raise a `RuntimeError` if earlier cells were skipped.
- **Google Colab vs. Local Storage Paths**:
  - In Google Colab (`IN_COLAB == True`), Cell 05 mounts Google Drive at `/content/drive` and sets `PERSISTENT_PROJECT_ROOT = Path("/content/drive/MyDrive/conformal_vlm_research")`.
  - Outside Google Colab (`IN_COLAB == False`), Cell 05 sets `MY_DRIVE = Path("/content")` and Cell 58 writes output ZIP archives to `/content/`. If running locally without root write permissions to `/content`, update `MY_DRIVE` in Cell 05 (Section 2) and `zip_path`, `summary_zip`, and `figures_zip` in Cell 58 (Section 26) to a writable local directory before running.

### Option B: Running the Synchronized Python Script

To run the standalone script from a terminal (after ensuring `/content` is writable or updating `MY_DRIVE` and the `/content/*.zip` paths):

```bash
python3 conformal_aware_selective_abstention_clip_colab_persistent_drive.py
```

### Required External Resources

On the first run (when `conformal_vlm_research/data/` and `conformal_vlm_research/cache/` are empty), the runtime must have outbound internet access to download:
1. OpenAI CLIP `ViT-B/32` weights (via `clip.load`).
2. `CIFAR-10`, `CIFAR-100`, and `SVHN` via `torchvision.datasets`.
3. `CIFAR-10-C.tar` (~2.9 GB) from Zenodo (`https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1`).

On subsequent runs against the same `conformal_vlm_research/` directory, existing datasets, model weights, and fingerprinted `.npz` embedding/logit caches are validated and reused without re-downloading or re-encoding images.

## 7. Inputs, Outputs, and Generated Files

### Inputs

- **Datasets**:
  - `CIFAR-10` train split (`50,000` RGB images, $32 \times 32$, 10 classes): used exclusively to sample the five 5,000-sample seed calibration splits (`CAL_SPLITS`).
  - `CIFAR-10` test split (`10,000` RGB images, $32 \times 32$, 10 classes): held-out in-distribution evaluation set (`E1`, `E2`) and ID reference set for OOD detection (`E7`–`E10`).
  - `CIFAR-10-C` (`gaussian_noise`, `motion_blur`, `fog`, `brightness`, `contrast`, `jpeg_compression` at severities `1`, `3`, `5`; `10,000` test images per corruption/severity slice): held-out visual shift evaluation sets (`E3`–`E6`).
  - `CIFAR-100` test split (`10,000` RGB images, $32 \times 32$): held-out semantic OOD evaluation set (`E7`, `E9`).
  - `SVHN` test split (`26,032` RGB images, $32 \times 32$): held-out far-OOD evaluation set (`E8`, `E10`).

### Generated Caches (`cache/`)

- `cache/text_embeddings/<prompt_key>_<fp>.npz` (and `.json`): `float32` array `text_embeddings` of shape `(10, 512)`.
- `cache/image_embeddings/<cache_name>_<fp>.npz` (and `.json`): `float32` array `image_embeddings` of shape `(N, 512)` and `int64` array `labels` of shape `(N,)`.
- `cache/logits/<cache_name>_<fp>.npz` (and `.json`): `float32` arrays `logits_P0`, `logits_P1`, `logits_P2`, and `logits_P3`, each of shape `(N, 10)`.

### Primary Output Files (`output/`)

- `output/all_results.csv` (`6,360` rows, `25` columns defined by `RESULT_COLUMNS`):
  - Columns: `seed`, `experiment_id`, `dataset`, `shift_type`, `shift_name`, `severity`, `prompt_condition`, `method`, `operating_rule`, `target_coverage`, `risk_target`, `threshold`, `coverage`, `risk`, `abstention_rate`, `aurc`, `eaurc`, `auroc`, `aupr_out`, `fpr95`, `ece`, `nll`, `brier`, `avg_set_size`, `score_orientation`.
  - Contains rows for `operating_rule` in `{"fixed_coverage", "risk_constraint", "curve_summary", "calibration", "ood_detection"}`.
- `output/per_sample_records.csv` (`9,961,920` rows, `19` columns):
  - Columns: `seed`, `experiment_id`, `dataset`, `shift_type`, `shift_name`, `severity`, `prompt_condition`, `condition_id`, `method`, `sample_id`, `true_label`, `predicted_label`, `confidence`, `uncertainty`, `prompt_agreement`, `prompt_instability`, `prompt_predictions`, `conformal_set_size`, `correctness`.

### Derived Tables (`tables/`)

- CSV tables (11 files): `main_method_comparison.csv`, `robustness_table.csv`, `risk_at_90_coverage.csv`, `risk_at_80_coverage.csv`, `risk_at_70_coverage.csv`, `risk_at_50_coverage.csv`, `risk_constrained_coverage.csv`, `calibration_metrics.csv`, `ood_metrics.csv`, `curve_metrics.csv`, `bootstrap_95ci.csv`, and `failure_cases.csv`.
- Markdown tables (8 files): `main_method_comparison.md`, `robustness_table.md`, `risk_at_90.md`, `risk_constrained.md`, `calibration_metrics.md`, `ood_metrics.md`, `bootstrap_95ci.md`, and `failure_cases.md`.

### Visualizations (`figures/`)

Each figure is saved as both `.png` (200 DPI) and `.pdf` (13 stems, 26 files total):
- `risk_coverage_E1_all`, `risk_coverage_E2_all`
- `risk_coverage_visual_<corruption>_s3` for each of the 6 corruptions in `CORRUPTIONS`
- `ood_auroc_comparison`
- `reliability_raw_vs_temperature`
- `prompt_agreement_vs_accuracy`
- `conformal_set_size_vs_correctness`
- `risk90_method_condition_heatmap`

### Metadata and Packaged Archives (`metadata/`, `final_output/`, `/content/`)

- `metadata/split_metadata.json`: Train/test sizes, seeded calibration indices, and indexing conventions.
- `metadata/seed_calibration_parameters.json`: Per-seed calibration indices, fitted temperatures, conformal quantiles, ranks, and quantile levels.
- `metadata/experiment_metadata.json`: Hardware/software versions, full configuration, expanded experiment matrix, and metric conventions.
- `metadata/final_validation_checklist.csv`: Pass/fail status of all 34 automated validation checks.
- `final_output/completion_summary.json` and `/content/conformal_vlm_research_artifacts.zip`, `/content/conformal_vlm_summary_tables.zip`, `/content/conformal_vlm_figures.zip`.

## 8. Implementation Details and Limitations

1. **Conformal Score Unscaled Probability Convention**: In `fit_seed_calibration` (Section 12) and `evaluate_seed_condition` (Section 16), split-conformal prediction sets and $\hat{q}_{1-\alpha}$ are computed from the unscaled base probabilities (`base["probs"]`), whereas temperature scaling is evaluated as a separate selective-prediction and OOD baseline (`temperature_MSP`).
2. **Discrete Tie Behavior for Coarse Uncertainty Scores**: `prompt_agreement` takes only four possible values in $\{0.25, 0.50, 0.75, 1.00\}$ (across 4 prompts) and `conformal_set_size` confidence $1 / |C(x)|$ takes at most 10 discrete values in $\{1/10, \dots, 1\}$. Because `selective_metrics` selects all samples satisfying `confidence >= threshold`, realized coverage at fixed nominal targets (such as 90%) can jump across discrete tie blocks.
3. **Seed Pooling in Visualization Helpers**: In Section 21 (Cell 48), calling `plot_risk_coverage("E1")` and `plot_risk_coverage("E2")` with `seed=None` (as well as the `E4` corruption curve loop) filters `sample_df` by experiment or condition without filtering by `seed`, pooling all 5 seeds when computing the plotted risk-coverage curves. Similarly, `prompt_agreement_vs_accuracy` and `conformal_set_size_vs_correctness` group `sample_df[sample_df.experiment_id == "E1"]` without filtering to a single `method` row per sample.
4. **Memory Footprint of `sample_df`**: `sample_df` holds $9,961,920$ rows in memory before writing `per_sample_records.csv`, requiring several gigabytes of RAM during Sections 18–26.
5. **Single-Process DataLoader Setting**: Section 10 overrides `CONFIG["num_workers"] = 0` to prevent multiprocessing child-process assertion errors in Google Colab notebooks.
