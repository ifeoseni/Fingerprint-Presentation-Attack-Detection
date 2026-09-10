# Fingerprint Presentation Attack Detection Using Support Vector Machine and K-Nearest Neighbor Classifiers

## Comprehensive Implementation Specification

## 1. Project Title

**Fingerprint Presentation Attack Detection Using Support Vector Machine and K-Nearest Neighbor Classifiers**

---

# 2. Project Objective

Develop a complete, reproducible machine learning system for fingerprint presentation attack detection using:

1. FVC2000 fingerprint databases.
2. SOCOFing fingerprint database.
3. Local Binary Pattern feature extraction.
4. Support Vector Machine classification.
5. K-Nearest Neighbor classification.

The objective is to determine whether a fingerprint image represents:

- **Genuine / Live fingerprint**
- **Spoof / Fake / Altered fingerprint**

The implementation must provide a scientifically reproducible experimental pipeline from raw dataset acquisition through preprocessing, feature extraction, model training, hyperparameter selection, testing, evaluation and result generation.

The system must not invent experimental values. Any value that is required by the final research report must be obtained from the actual implementation and stored in the experiment results.

---

# 3. Primary Research Questions

The implementation should be designed to answer the following questions.

### RQ1

How accurately can Local Binary Pattern features distinguish genuine fingerprints from spoofed or altered fingerprints?

### RQ2

Which performs better for fingerprint presentation attack detection:

- Support Vector Machine
- K-Nearest Neighbor

### RQ3

How well do the models perform when evaluated against difficult fingerprint alterations?

### RQ4

Does the performance of the classifiers vary between FVC2000 acquisition technologies and SOCOFing altered fingerprints?

### RQ5

How robust are the classifiers when tested against the **SOCOFing Altered-Hard** subset?

---

# 4. Datasets

The experiment shall use exactly two principal data sources:

```text
Dataset 1: FVC2000
Dataset 2: SOCOFing
```

Do not introduce another dataset unless explicitly instructed.

---

# 5. FVC2000 Dataset

The FVC2000 source shall be obtained from the supplied Springer supplementary material.

The supplied FVC2000 report identifies four databases:

| Database | Technology |
|---|---|
| DB1 | KeyTronic Secure Desktop Scanner |
| DB2 | ST Microelectronics TouchChip |
| DB3 | Identicator Technology DF-90 |
| DB4 | Synthetic fingerprint generation |

The report states that each database contains 110 fingers and 8 impressions per finger, giving 880 fingerprints per database. Fingers 101 to 110 constitute Set B for parameter tuning, while fingers 1 to 100 constitute Set A for the benchmark.

The implementation must preserve:

```text
FVC2000
├── DB1
├── DB2
├── DB3
└── DB4
```

and, where the source structure permits:

```text
DBx
├── Set A
└── Set B
```

Do not combine the four databases without retaining database metadata.

---

# 6. FVC2000 Experimental Role

FVC2000 should primarily be used as the **genuine fingerprint source and technology-diverse evaluation source**.

The implementation must first inspect the actual supplementary archive and determine exactly what image types and labels are available.

Do not assume that every FVC2000 image is a genuine/spoof pair.

This is important because the supplied FVC2000 report is fundamentally a fingerprint recognition benchmark and describes genuine and impostor comparisons rather than a modern presentation attack dataset. The report itself cautions that FVC2000 is a technology evaluation rather than an official real-world biometric certification.

Therefore:

**Do not automatically label FVC2000 impostor fingerprints as physical presentation attacks.**

The implementation must distinguish:

```text
genuine fingerprint
```

from:

```text
different-person/impostor fingerprint
```

and:

```text
actual altered/spoof fingerprint
```

where the source data support that distinction.

If FVC2000 does not contain presentation attack images suitable for the intended binary PAD experiment, document that limitation instead of fabricating spoof labels.

---

# 7. SOCOFing Dataset

Use:

**SOCOFing: Sokoto Coventry Fingerprint Dataset**

Source:

https://www.kaggle.com/datasets/ruizgara/socofing

The implementation must inspect the downloaded dataset and identify:

```text
Real/
Altered/
```

and:

```text
Altered-Easy/
Altered-Medium/
Altered-Hard/
```

---

# 8. SOCOFing Class Definition

The primary SOCOFing experiment shall use:

### Genuine

```text
Real/
```

Label:

```text
0 = Genuine
```

### Spoof/Altered

For the principal PAD experiment:

```text
Altered/Altered-Hard/
```

Label:

```text
1 = Spoof/Altered
```

This means the primary SOCOFing experiment is:

```text
Real
      VS
Altered-Hard
```

This is the principal test condition.

---

# 9. SOCOFing Easy and Medium

Easy and Medium must not be discarded from the project.

They should be retained for secondary robustness experiments.

The implementation should support:

```text
Experiment A:
Real vs Altered-Easy

Experiment B:
Real vs Altered-Medium

Experiment C:
Real vs Altered-Hard
```

Experiment C is the **primary experiment**.

This allows the final research to demonstrate how classifier performance changes as alteration difficulty increases.

---

# 10. SOCOFing Experimental Matrix

Run:

| Experiment | Genuine | Altered | Purpose |
|---|---|---|---|
| SOCOFing-Easy | Real | Altered-Easy | Baseline difficulty |
| SOCOFing-Medium | Real | Altered-Medium | Intermediate difficulty |
| SOCOFing-Hard | Real | Altered-Hard | Primary difficult PAD evaluation |

For each experiment run:

```text
SVM + LBP
KNN + LBP
```

Therefore the minimum SOCOFing experiment count is:

```text
3 difficulty levels
×
2 classifiers
=
6 experiments
```

---

# 11. Dataset Combination

Do not immediately merge FVC2000 and SOCOFing.

First perform independent experiments:

```text
FVC2000 -> SVM
FVC2000 -> KNN

SOCOFing -> SVM
SOCOFing -> KNN
```

The combined dataset experiment should be optional and only performed if the data semantics permit meaningful combination.

Do not combine:

```text
FVC2000 impostor
```

with:

```text
SOCOFing altered
```

and call both "spoof" without explicitly documenting the semantic difference.

---

# 12. Dataset Audit

Before training any model, create an automated dataset audit.

The audit must report:

```text
Total images
Number of genuine images
Number of altered images
Number of images per dataset
Number of images per FVC database
Number of images per SOCOFing alteration level
Number of images per hand
Number of images per finger
Number of images per identity
```

Generate:

```text
dataset_statistics.csv
dataset_statistics.json
```

Example:

```text
Dataset              Class              Count
------------------------------------------------
FVC2000 DB1          Genuine            X
FVC2000 DB2          Genuine            X
FVC2000 DB3          Genuine            X
FVC2000 DB4          Genuine            X
SOCOFing             Genuine            X
SOCOFing             Altered-Easy       X
SOCOFing             Altered-Medium     X
SOCOFing             Altered-Hard       X
```

The program must calculate these values.

Never manually enter them.

---

# 13. Dataset Metadata

Create a metadata table with at least:

```text
image_id
file_path
dataset
database
subset
class
class_label
finger_id
hand
finger_position
alteration_level
original_filename
```

Additional fields may be added if they can be reliably extracted.

For example:

```text
dataset = SOCOFing
class = genuine
class_label = 0
alteration_level = none
```

or:

```text
dataset = SOCOFing
class = altered
class_label = 1
alteration_level = hard
```

---

# 14. File Validation

Before processing an image:

1. Verify that the file exists.
2. Verify that it is readable.
3. Verify that it can be decoded.
4. Verify that the image contains valid pixel values.
5. Record corrupted images.
6. Do not silently discard corrupted images.

Create:

```text
invalid_images.csv
```

containing:

```text
file_path
error
dataset
```

---

# 15. Duplicate Detection

Implement duplicate detection.

At minimum calculate:

```text
SHA-256 file hash
```

and optionally:

```text
perceptual hash
```

to identify exact or near-duplicate images.

Duplicates must not result in the same fingerprint identity appearing in both training and testing datasets.

---

# 16. Identity-Aware Dataset Splitting

This is a mandatory requirement.

Do not randomly split individual fingerprint images if multiple images belong to the same finger.

For example, this is prohibited:

```text
Finger 001 impression 1 -> training
Finger 001 impression 2 -> testing
```

because it creates identity leakage.

Instead:

```text
Finger 001 -> training
Finger 002 -> training
...
Finger 090 -> training

Finger 091 -> testing
...
```

The exact identities must be selected reproducibly.

---

# 17. Training/Test Split

Use an 80:20 split for the principal experiment where the dataset structure permits it.

The split should be:

```text
80% -> training
20% -> testing
```

and should be:

```text
identity-aware
class-balanced where possible
reproducible
```

Use:

```text
random_state = 42
```

unless the experiment requires another explicitly documented seed.

The actual random seed must be stored in the experiment configuration.

---

# 18. Cross-Validation

Hyperparameter selection must occur only within the training data.

Use:

```text
5-fold stratified cross-validation
```

where ordinary class stratification is appropriate.

If identity grouping is necessary, use an appropriate group-aware cross-validation strategy instead.

The final test set must remain untouched.

Correct:

```text
Full dataset
    |
    +---- Training set
    |        |
    |        +---- Cross-validation
    |                 |
    |                 +---- Hyperparameter tuning
    |
    +---- Test set
             |
             +---- Final evaluation
```

Incorrect:

```text
Full dataset
    |
    +---- Cross-validation
    |
    +---- Test
```

when the test data have already influenced parameter selection.

---

# 19. Image Preprocessing Pipeline

Every image must pass through the same documented pipeline.

```text
Raw fingerprint
       |
       v
Image validation
       |
       v
Grayscale conversion
       |
       v
64 x 64 resizing
       |
       v
Pixel normalization
       |
       v
Noise reduction
       |
       v
LBP extraction
       |
       v
Histogram generation
       |
       v
Feature normalization
       |
       v
Classifier
```

---

# 20. Grayscale Conversion

All images must be converted to grayscale before LBP extraction.

If an image is already grayscale, preserve it.

Do not introduce unnecessary colour processing.

---

# 21. Image Resizing

Resize every fingerprint to:

```text
64 x 64 pixels
```

Use nearest-neighbor interpolation to match the existing methodology.

Document:

```text
width = 64
height = 64
interpolation = nearest
```

---

# 22. Pixel Normalization

Convert image pixels to floating-point values.

For standard 8-bit images:

```text
normalized_pixel = pixel / 255
```

The resulting range must be:

```text
0 to 1
```

Verify this programmatically.

---

# 23. Noise Reduction

The implementation must reproduce the intended denoising methodology as accurately as possible.

The existing draft states:

> Averaging of multiple noisy images was used to reduce noise.

The AI agent must not implement arbitrary averaging without determining whether corresponding multiple impressions are actually available and spatially compatible.

The agent must therefore:

1. Inspect the source dataset.
2. Determine whether multiple corresponding impressions are available.
3. Determine whether they can legitimately be averaged.
4. If multi-image averaging is not scientifically justified, implement a documented per-image denoising method.
5. Record the selected method in the experiment configuration.

Do not claim multi-image averaging in the final report unless it was actually performed.

---

# 24. LBP Feature Extraction

Use:

```python
skimage.feature.local_binary_pattern
```

The implementation must make the following parameters configurable:

```text
P = number of sampling points
R = radius
method = pattern type
```

Baseline configuration:

```text
P = 8
R = 1
method = uniform
```

The agent must record the exact configuration used.

---

# 25. LBP Calculation

For every image:

```text
64 x 64 normalized image
        |
        v
LBP transformation
        |
        v
LBP code image
```

Then generate a normalized histogram.

The histogram should represent the distribution of local texture patterns.

---

# 26. LBP Histogram Normalization

Calculate:

```text
H_normalized = H / sum(H)
```

where:

```text
H = LBP histogram
```

The implementation must verify that the histogram sums approximately to 1.

---

# 27. LBP Feature Dimension

The implementation must automatically determine and record:

```text
LBP feature vector dimension
```

Do not guess the dimension.

The final methodology must use the dimension produced by the actual implementation.

---

# 28. Optional Spatial LBP

Implement spatial LBP as an optional experimental configuration.

For example:

```text
4 x 4 image blocks
```

Extract one histogram per block and concatenate the histograms.

However, the principal baseline should remain:

```text
Global LBP
P = 8
R = 1
uniform
```

unless experiments demonstrate that another configuration is more appropriate.

---

# 29. Feature Scaling

The feature scaler must be fitted exclusively on training data.

Correct:

```text
fit scaler on X_train
transform X_train
transform X_test
```

Never:

```text
fit scaler on X_train + X_test
```

This is a mandatory leakage prevention rule.

---

# 30. Support Vector Machine

Implement SVM using:

```python
sklearn.svm.SVC
```

The primary kernel shall be:

```text
RBF
```

Expose:

```text
kernel
C
gamma
class_weight
probability
```

---

# 31. SVM Hyperparameter Search

Use cross-validation.

Search:

```text
C:
0.1
1
10
100

gamma:
scale
0.001
0.01
0.1
1
```

The agent may expand this grid if computationally feasible.

The final model must record:

```text
best_kernel
best_C
best_gamma
best_class_weight
```

Do not manually select parameters because they are commonly used.

---

# 32. K-Nearest Neighbor

Implement:

```python
sklearn.neighbors.KNeighborsClassifier
```

Use Euclidean distance as the primary metric.

Baseline:

```text
metric = euclidean
weights = uniform
```

Expose:

```text
n_neighbors
metric
weights
```

---

# 33. KNN Hyperparameter Search

Test:

```text
K = 1
K = 3
K = 5
K = 7
K = 9
K = 11
K = 15
K = 21
```

Use cross-validation on the training data.

The selected value must be recorded automatically.

Example:

```text
best_K = 7
```

must only appear if the actual experiment selects K = 7.

---

# 34. Class Imbalance

Before training, calculate:

```text
genuine_count
spoof_count
class_ratio
```

If classes are imbalanced, do not simply remove data to force balance.

The implementation should evaluate:

1. Original distribution.
2. Class-weighted SVM where appropriate.
3. Appropriate KNN configuration.
4. Stratified evaluation.

Any balancing method used must be documented.

---

# 35. Primary SOCOFing Experiment

This is the most important experiment.

Use:

```text
Genuine:
SOCOFing/Real/

Spoof:
SOCOFing/Altered/Altered-Hard/
```

The experiment shall answer:

> How effectively can LBP-based SVM and KNN distinguish genuine fingerprints from difficult altered fingerprints?

Run:

```text
SOCOFing Hard + LBP + SVM
SOCOFing Hard + LBP + KNN
```

---

# 36. SOCOFing Difficulty Analysis

After the primary Hard experiment, run:

```text
Real vs Altered-Easy
Real vs Altered-Medium
Real vs Altered-Hard
```

for both models.

Create:

```text
SOCOFing Difficulty
        |
        +---- Easy
        +---- Medium
        +---- Hard
```

Plot:

```text
Accuracy vs difficulty
Precision vs difficulty
Recall vs difficulty
F1-score vs difficulty
```

This provides evidence of classifier robustness as alteration difficulty increases.

---

# 37. FVC2000 Experiments

Because FVC2000 contains multiple acquisition technologies, run experiments separately where the available labels and experimental objective support them.

At minimum retain separate metadata and results for:

```text
DB1
DB2
DB3
DB4
```

The report identifies DB1 and DB2 as easier than DB3 in its original recognition benchmark and states that DB4 is synthetically generated.

Do not interpret those original recognition results as PAD results.

Our SVM/KNN results must be independently generated by our implementation.

---

# 38. FVC2000 DB-Level Evaluation

Generate:

```text
FVC_DB1_SVM
FVC_DB1_KNN

FVC_DB2_SVM
FVC_DB2_KNN

FVC_DB3_SVM
FVC_DB3_KNN

FVC_DB4_SVM
FVC_DB4_KNN
```

where the data labels and experiment design support classification.

Also generate:

```text
FVC2000 average SVM
FVC2000 average KNN
```

only after defining precisely how the average is calculated.

---

# 39. Avoiding Misinterpretation of FVC2000

The AI agent must include a research note:

FVC2000 was originally designed as a fingerprint recognition technology evaluation rather than as a dedicated presentation attack detection database. The original report states that the competition evaluated algorithms using images from sensors not native to each system and was not intended as official certification of real-world biometric systems.

Therefore:

```text
FVC2000 recognition/impostor data
```

must not automatically be represented as:

```text
physical spoof attacks
```

unless the specific supplementary data explicitly establish this.

This distinction is mandatory for the academic integrity of the project.

---

# 40. Evaluation Metrics

Calculate:

### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1-score

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

Calculate all four metrics for both classifiers.

---

# 41. Confusion Matrix

For every experiment produce:

```text
Confusion matrix
```

with:

```text
                 Predicted
                 Genuine Spoof

Actual Genuine     TN      FP

Actual Spoof       FN      TP
```

The exact positive-class convention must be documented.

Recommended:

```text
0 = Genuine
1 = Spoof
```

Therefore:

```text
Positive class = Spoof
```

This makes precision, recall and F1 directly relevant to attack detection.

---

# 42. Classification Report

Generate a complete classification report containing:

```text
precision
recall
f1-score
support
```

for:

```text
Genuine
Spoof
```

Also report:

```text
macro average
weighted average
accuracy
```

---

# 43. Additional PAD Metrics

Where the experiment supports meaningful calculation, also calculate:

```text
False Acceptance Rate
False Rejection Rate
Attack Presentation Classification Error Rate
```

Do not introduce these metrics without clearly defining the decision convention.

---

# 44. SVM Decision Scores

For SVM, save:

```text
decision_function
```

for the test samples.

If probability estimation is enabled, clearly distinguish:

```text
decision score
```

from:

```text
probability
```

Do not call a raw SVM decision score a probability.

---

# 45. KNN Probability

Where enabled, save KNN class probabilities.

This may be used for:

```text
ROC
AUC
threshold analysis
```

but only if the probability estimates are appropriately interpreted.

---

# 46. ROC and AUC

Although accuracy, precision, recall and F1-score are mandatory, implement optional:

```text
ROC curve
AUC
```

for both models where continuous decision/probability scores are available.

Generate:

```text
ROC_SVM.png
ROC_KNN.png
```

and optionally:

```text
ROC_SVM_vs_KNN.png
```

---

# 47. Precision-Recall Curve

For imbalanced datasets, generate:

```text
Precision-Recall curve
```

where appropriate.

This is particularly useful for SOCOFing if the genuine and altered class counts differ substantially.

---

# 48. Results Table

Generate:

| Dataset | Difficulty | Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| SOCOFing | Easy | SVM | | | | |
| SOCOFing | Easy | KNN | | | | |
| SOCOFing | Medium | SVM | | | | |
| SOCOFing | Medium | KNN | | | | |
| SOCOFing | Hard | SVM | | | | |
| SOCOFing | Hard | KNN | | | | |

The Hard rows constitute the primary SOCOFing results.

---

# 49. FVC Results Table

Generate:

| Database | Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| DB1 | SVM | | | | |
| DB1 | KNN | | | | |
| DB2 | SVM | | | | |
| DB2 | KNN | | | | |
| DB3 | SVM | | | | |
| DB3 | KNN | | | | |
| DB4 | SVM | | | | |
| DB4 | KNN | | | | |

Only populate rows corresponding to valid experiments.

---

# 50. Classifier Comparison

Create direct SVM versus KNN comparisons.

For each dataset calculate:

```text
Accuracy difference
Precision difference
Recall difference
F1 difference
```

Example:

```text
SVM F1 - KNN F1
```

This provides an objective basis for identifying the better classifier.

---

# 51. Statistical Comparison

Where computationally feasible, perform repeated evaluation using multiple random seeds or appropriate cross-validation.

For example:

```text
seed = 42
seed = 43
seed = 44
seed = 45
seed = 46
```

Do not repeatedly tune against the final test set.

Report:

```text
mean
standard deviation
```

for the repeated training/evaluation experiments.

If the final study uses only one fixed hold-out test set, state this clearly rather than implying statistical repetition.

---

# 52. Reproducibility Configuration

Create:

```text
configs/experiment.yaml
```

Example:

```yaml
project:
  name: fingerprint_pad
  random_state: 42

image:
  width: 64
  height: 64
  interpolation: nearest
  grayscale: true

normalization:
  method: min_max
  range:
    - 0
    - 1

lbp:
  points: 8
  radius: 1
  method: uniform
  spatial: false

split:
  test_size: 0.20
  strategy: identity_aware
  random_state: 42

cross_validation:
  folds: 5

svm:
  kernel: rbf
  C:
    - 0.1
    - 1
    - 10
    - 100
  gamma:
    - scale
    - 0.001
    - 0.01
    - 0.1
    - 1

knn:
  neighbors:
    - 1
    - 3
    - 5
    - 7
    - 9
    - 11
    - 15
    - 21
  metric: euclidean
  weights: uniform

datasets:
  socofing:
    primary_attack_level: hard
```

---

# 53. Automatic Environment Capture

The implementation must record:

```text
Operating system
Python version
NumPy version
Pandas version
scikit-learn version
scikit-image version
OpenCV version
Matplotlib version
Joblib version
CPU
CPU cores
RAM
GPU if available
```

Save:

```text
results/system_information.json
```

Also generate:

```text
requirements.txt
```

and:

```text
requirements-lock.txt
```

using the actual installed environment.

---

# 54. Project Directory

Use:

```text
fingerprint-pad/
│
├── data/
│   ├── raw/
│   │   ├── fvc2000/
│   │   └── socofing/
│   │
│   ├── processed/
│   └── metadata/
│
├── configs/
│   └── experiment.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset_audit.py
│   ├── dataset_loader.py
│   ├── metadata.py
│   ├── validation.py
│   ├── duplicate_detection.py
│   ├── preprocessing.py
│   ├── denoising.py
│   ├── lbp_features.py
│   ├── splitting.py
│   ├── scaling.py
│   ├── svm_classifier.py
│   ├── knn_classifier.py
│   ├── hyperparameter_tuning.py
│   ├── evaluation.py
│   ├── plots.py
│   ├── experiment_runner.py
│   └── system_info.py
│
├── scripts/
│   ├── audit_dataset.py
│   ├── preprocess_dataset.py
│   ├── extract_features.py
│   ├── train_svm.py
│   ├── train_knn.py
│   ├── evaluate.py
│   └── run_all_experiments.py
│
├── features/
│   ├── fvc2000/
│   └── socofing/
│
├── models/
│   ├── svm/
│   └── knn/
│
├── results/
│   ├── metrics/
│   ├── tables/
│   ├── confusion_matrices/
│   ├── roc/
│   ├── precision_recall/
│   ├── preprocessing_examples/
│   └── logs/
│
├── tests/
│   ├── test_dataset_loader.py
│   ├── test_preprocessing.py
│   ├── test_lbp.py
│   ├── test_splitting.py
│   ├── test_scaling.py
│   ├── test_svm.py
│   ├── test_knn.py
│   └── test_evaluation.py
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_lbp_features.ipynb
│   ├── 04_svm.ipynb
│   ├── 05_knn.ipynb
│   └── 06_results_analysis.ipynb
│
├── README.md
├── requirements.txt
├── requirements-lock.txt
└── run_experiment.py
```

---

# 55. Code Architecture

The implementation must be modular.

Do not place the entire project in one Python file.

The following separation is mandatory:

```text
Dataset loading
        !=
Preprocessing
        !=
Feature extraction
        !=
Model training
        !=
Evaluation
        !=
Visualization
```

Each module must have clear functions and documentation.

---

# 56. Dataset Loader

Implement functions such as:

```python
load_fvc2000()
load_socofing()
load_socofing_real()
load_socofing_altered_easy()
load_socofing_altered_medium()
load_socofing_altered_hard()
```

Each function should return structured metadata rather than only a list of image paths.

---

# 57. Preprocessing Function

Implement a function similar to:

```python
preprocess_image(image, config)
```

It should:

1. Validate the image.
2. Convert to grayscale.
3. Resize to 64 x 64.
4. Normalize pixels.
5. Apply the configured denoising method.
6. Return the processed image.

---

# 58. Feature Extraction Function

Implement:

```python
extract_lbp_features(image, config)
```

It must:

1. Calculate LBP.
2. Calculate histogram.
3. Normalize histogram.
4. Return a one-dimensional feature vector.

---

# 59. SVM Pipeline

Use a reproducible pipeline:

```text
Features
   |
   v
Scaler
   |
   v
SVM
```

Where hyperparameter tuning is required:

```text
Pipeline
   |
   v
GridSearchCV
   |
   v
Best estimator
```

---

# 60. KNN Pipeline

Use:

```text
Features
   |
   v
Scaler
   |
   v
KNN
```

and tune K only using the training set.

---

# 61. Model Saving

Save the complete fitted pipeline.

Examples:

```text
models/svm/socofing_hard_svm.joblib
models/knn/socofing_hard_knn.joblib
```

Also save:

```text
models/svm/fvc_db1_svm.joblib
models/knn/fvc_db1_knn.joblib
```

as applicable.

---

# 62. Experiment Naming

Use consistent experiment IDs.

Examples:

```text
SOCOFING_HARD_SVM_LBP
SOCOFING_HARD_KNN_LBP

SOCOFING_MEDIUM_SVM_LBP
SOCOFING_MEDIUM_KNN_LBP

SOCOFING_EASY_SVM_LBP
SOCOFING_EASY_KNN_LBP

FVC_DB1_SVM_LBP
FVC_DB1_KNN_LBP
```

Every output must contain its experiment ID.

---

# 63. Experiment Manifest

Each experiment must generate:

```text
experiment_manifest.json
```

Example:

```json
{
  "experiment_id": "SOCOFING_HARD_SVM_LBP",
  "dataset": "SOCOFing",
  "genuine_source": "Real",
  "spoof_source": "Altered-Hard",
  "classifier": "SVM",
  "feature": "LBP",
  "image_size": [64, 64],
  "lbp_points": 8,
  "lbp_radius": 1,
  "lbp_method": "uniform",
  "test_size": 0.2,
  "random_state": 42,
  "cross_validation_folds": 5
}
```

After training, append:

```text
best_parameters
training_count
testing_count
accuracy
precision
recall
f1_score
```

---

# 64. Results Storage

Save results in machine-readable form.

Required:

```text
results/metrics/final_results.csv
results/metrics/final_results.json
```

The CSV must contain:

```text
experiment_id
dataset
database
difficulty
classifier
feature
train_count
test_count
accuracy
precision
recall
f1_score
best_parameters
```

---

# 65. Visualization Requirements

Generate:

1. Dataset class distribution.
2. Dataset distribution by source.
3. Genuine versus altered distribution.
4. Original fingerprint examples.
5. Preprocessed fingerprint examples.
6. Denoised fingerprint examples.
7. LBP image examples.
8. LBP histogram.
9. SVM confusion matrices.
10. KNN confusion matrices.
11. SVM versus KNN accuracy.
12. SVM versus KNN F1-score.
13. SOCOFing Easy versus Medium versus Hard.
14. ROC curves where appropriate.
15. Precision-recall curves where appropriate.

Use publication-quality resolution.

---

# 66. Required Preprocessing Figure

Generate a figure showing:

```text
Original image
        |
        v
64 x 64 image
        |
        v
Normalized image
        |
        v
Denoised image
        |
        v
LBP representation
```

The figure should be suitable for inclusion in the methodology chapter.

---

# 67. Required SOCOFing Hard Figure

Generate a figure showing examples of:

```text
Genuine SOCOFing fingerprint
```

and:

```text
Altered-Hard fingerprint
```

Do not alter the examples merely to make the distinction visually obvious.

---

# 68. Experiment Execution

The complete experiment must be executable using one command.

For example:

```bash
python run_experiment.py
```

The command should:

```text
1. Validate datasets
2. Audit datasets
3. Build metadata
4. Create reproducible splits
5. Preprocess images
6. Extract LBP features
7. Tune SVM
8. Tune KNN
9. Train final models
10. Evaluate models
11. Generate metrics
12. Generate figures
13. Save models
14. Save experiment manifests
15. Save environment information
```

---

# 69. Separate Execution Commands

Also provide individual commands:

```bash
python scripts/audit_dataset.py
```

```bash
python scripts/preprocess_dataset.py
```

```bash
python scripts/extract_features.py
```

```bash
python scripts/train_svm.py
```

```bash
python scripts/train_knn.py
```

```bash
python scripts/evaluate.py
```

```bash
python scripts/run_all_experiments.py
```

---

# 70. Testing Requirements

Implement unit tests before accepting the experiment.

Test:

### Dataset tests

- Dataset directories are detected.
- Images are readable.
- Metadata is generated.
- Labels are correct.
- Altered-Hard images are correctly identified.

### Preprocessing tests

- Output is 64 x 64.
- Output is grayscale.
- Pixel values are within 0 to 1.

### LBP tests

- LBP transformation succeeds.
- Histogram is produced.
- Histogram is normalized.
- Feature dimensions are consistent.

### Split tests

- No identity overlap.
- Training and testing sets are disjoint.
- Class labels are valid.

### Model tests

- SVM trains.
- KNN trains.
- Predictions have the correct shape.

### Evaluation tests

- Accuracy is between 0 and 1.
- Precision is between 0 and 1.
- Recall is between 0 and 1.
- F1-score is between 0 and 1.

---

# 71. Leakage Detection

The program must explicitly test for:

```text
Training/test identity overlap
Training/test filename overlap
Training/test hash overlap
Scaler leakage
Hyperparameter tuning leakage
```

If leakage is detected:

```text
STOP EXPERIMENT
```

Do not continue and report the result.

---

# 72. Reproducibility Test

After completing the first experiment, rerun it using the same:

```text
dataset
configuration
random seed
software environment
```

The result should be identical or numerically equivalent within documented floating-point tolerances.

Save:

```text
results/reproducibility_check.json
```

---

# 73. Final Model Selection

The final classifier must be selected based on the predefined evaluation metrics.

Do not select a model merely because it produces the highest accuracy after inspecting multiple test runs.

The preferred primary metric should be:

```text
F1-score
```

because the task involves distinguishing genuine samples from attacks and may involve class imbalance.

Accuracy, precision and recall must still be reported.

---

# 74. Primary Comparison

The most important comparison is:

```text
SOCOFing Altered-Hard

LBP + SVM
        VS
LBP + KNN
```

Report:

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
```

The final report should explicitly identify which classifier performs better under the Hard condition.

---

# 75. Difficulty Analysis

The second major comparison is:

```text
             SVM              KNN

Easy         result           result

Medium       result           result

Hard         result           result
```

Analyse whether performance:

```text
increases
decreases
remains stable
```

as alteration difficulty increases.

Do not assume Hard will produce lower accuracy. Report what the experiment actually demonstrates.

---

# 76. FVC2000 Analysis

Analyse performance across:

```text
DB1
DB2
DB3
DB4
```

where valid classification experiments can be constructed.

The original FVC2000 report notes differences in recognition difficulty across these databases, including DB1 and DB2 being easier than DB3 and DB4 being synthetically generated.

Our experiment must not simply reproduce those original benchmark results. It must generate independent SVM and KNN results.

---

# 77. Academic Integrity Requirements

The implementation must never:

- Invent results.
- Invent dataset counts.
- Invent hyperparameters.
- Invent hardware specifications.
- Invent software versions.
- Claim cross-validation if it was not performed.
- Claim a spoof sample is physical material when the dataset does not establish this.
- Remove difficult samples to increase performance.
- Tune on the final test set.
- Report test performance before the complete experimental protocol is fixed.
- Modify the methodology merely to obtain a better result.

---

# 78. Final Report Data Extraction

The implementation must automatically produce a file:

```text
results/report_values.json
```

containing every numerical value required for the academic manuscript.

Example:

```json
{
  "dataset": {
    "fvc2000_total": null,
    "socofing_total": null,
    "socofing_real": null,
    "socofing_altered_easy": null,
    "socofing_altered_medium": null,
    "socofing_altered_hard": null
  },
  "preprocessing": {
    "image_width": 64,
    "image_height": 64,
    "interpolation": "nearest",
    "normalization": "0-1"
  },
  "lbp": {
    "points": 8,
    "radius": 1,
    "method": "uniform",
    "feature_dimension": null
  },
  "split": {
    "ratio": "80:20",
    "random_state": 42,
    "identity_aware": true
  },
  "svm": {
    "kernel": "rbf",
    "C": null,
    "gamma": null
  },
  "knn": {
    "K": null,
    "metric": "euclidean"
  }
}
```

The null values must be populated automatically after the experiments are executed.

---

# 79. Methodology Statement Generation

After all experiments have completed, create:

```text
results/methodology_values.md
```

containing prose-ready factual statements derived from the actual implementation.

For example:

```text
The images were resized to 64 × 64 pixels using nearest-neighbor interpolation.

LBP features were extracted using P = X sampling points and R = X.

The final LBP feature vector contained X features.

The dataset was divided into training and testing subsets using an identity-aware 80:20 split.

SVM hyperparameters were selected using five-fold cross-validation.

The optimal SVM configuration was C = X and gamma = X.

The optimal KNN configuration used K = X.
```

These statements must be generated from the experiment configuration and results.

---

# 80. Final Deliverables

The AI agent must deliver:

```text
Complete source code
Dataset loader
Dataset audit
Metadata generator
Preprocessing pipeline
Denoising implementation
LBP implementation
SVM implementation
KNN implementation
Hyperparameter tuning
Identity-aware splitting
Evaluation system
Confusion matrices
ROC curves where applicable
Precision-recall curves where applicable
Comparison plots
Trained models
Experiment manifests
Environment information
Unit tests
Requirements files
README
Final CSV results
Final JSON results
Methodology values
```

---

# 81. Final Expected Directory

The final project should look approximately like:

```text
fingerprint-pad/
│
├── data/
│   ├── raw/
│   │   ├── fvc2000/
│   │   └── socofing/
│   ├── processed/
│   └── metadata/
│
├── configs/
│   └── experiment.yaml
│
├── features/
│   ├── fvc2000/
│   └── socofing/
│
├── models/
│   ├── svm/
│   └── knn/
│
├── src/
│   ├── dataset_loader.py
│   ├── dataset_audit.py
│   ├── metadata.py
│   ├── validation.py
│   ├── duplicate_detection.py
│   ├── preprocessing.py
│   ├── denoising.py
│   ├── lbp_features.py
│   ├── splitting.py
│   ├── scaling.py
│   ├── svm_classifier.py
│   ├── knn_classifier.py
│   ├── hyperparameter_tuning.py
│   ├── evaluation.py
│   ├── plots.py
│   ├── experiment_runner.py
│   └── system_info.py
│
├── scripts/
│   ├── audit_dataset.py
│   ├── preprocess_dataset.py
│   ├── extract_features.py
│   ├── train_svm.py
│   ├── train_knn.py
│   ├── evaluate.py
│   └── run_all_experiments.py
│
├── results/
│   ├── metrics/
│   ├── tables/
│   ├── confusion_matrices/
│   ├── roc/
│   ├── precision_recall/
│   ├── plots/
│   ├── logs/
│   ├── experiment_manifests/
│   ├── system_information.json
│   ├── report_values.json
│   └── methodology_values.md
│
├── tests/
│
├── notebooks/
│
├── requirements.txt
├── requirements-lock.txt
├── README.md
└── run_experiment.py
```

---

# 82. Definition of Done

The implementation is considered complete only when all of the following are true:

```text
[ ] FVC2000 source has been inspected and documented.
[ ] SOCOFing source has been inspected and documented.
[ ] Exact dataset counts have been calculated.
[ ] Genuine and altered classes have been verified.
[ ] Altered-Hard is used as the primary SOCOFing attack condition.
[ ] Easy and Medium are retained for secondary analysis.
[ ] Images are resized to 64 × 64.
[ ] Pixel normalization is implemented.
[ ] Denoising methodology is documented.
[ ] LBP parameters are explicitly recorded.
[ ] LBP feature dimension is calculated automatically.
[ ] Identity-aware splitting is implemented.
[ ] 80:20 split is reproducible.
[ ] Test data are isolated.
[ ] Five-fold cross-validation is used for hyperparameter selection.
[ ] SVM RBF implementation is complete.
[ ] SVM C and gamma are selected experimentally.
[ ] KNN implementation is complete.
[ ] KNN K is selected experimentally.
[ ] Accuracy is calculated.
[ ] Precision is calculated.
[ ] Recall is calculated.
[ ] F1-score is calculated.
[ ] Confusion matrices are generated.
[ ] SOCOFing Easy experiment is complete.
[ ] SOCOFing Medium experiment is complete.
[ ] SOCOFing Hard experiment is complete.
[ ] FVC2000 database-level experiments are complete where scientifically valid.
[ ] SVM and KNN are compared.
[ ] Models are saved.
[ ] Experiment configurations are saved.
[ ] Environment information is saved.
[ ] Unit tests pass.
[ ] Leakage tests pass.
[ ] Reproducibility test passes.
[ ] Final results are exported.
[ ] Methodology values are automatically generated.
[ ] No numerical result in the report is manually invented.
```

# 83. Final Instruction to the AI Coding Agent

**Implement the entire project according to this specification.**

Do not begin model training until the dataset audit has completed.

Do not assume dataset structure until the downloaded files have been inspected.

Do not invent labels.

Do not interpret FVC2000 impostor fingerprints as presentation attacks without evidence from the source data.

Use **SOCOFing Real versus Altered-Hard as the principal PAD experiment**.

Use Easy and Medium as secondary difficulty experiments.

Maintain complete metadata throughout the pipeline.

Prevent identity leakage.

Use the test set only for final evaluation.

Automatically record all parameters and environment information.

Automatically calculate all dataset counts, feature dimensions, hyperparameters and evaluation metrics.

The final implementation must be executable from a clean environment using the documented commands.

The final research report must be reproducible from the generated configuration, source code, dataset metadata and experiment outputs.

**Most importantly, never manufacture a result simply to fill a placeholder in the manuscript. If an expected experiment cannot legitimately be performed using the supplied data, stop that experiment, document the reason, and report the limitation.**