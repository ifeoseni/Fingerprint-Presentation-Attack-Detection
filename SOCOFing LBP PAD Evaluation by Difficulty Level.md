# SOCOFing LBP PAD Evaluation by Difficulty Level

## Objective

Extend the existing fingerprint Presentation Attack Detection pipeline so that the LBP + SVM and LBP + K-nearest neighbor experiments are evaluated separately on:

1. Altered-Easy
2. Altered-Medium
3. Altered-Hard
4. All altered categories combined

The purpose is to determine how the difficulty of the fingerprint alteration affects the performance of the LBP-based Presentation Attack Detection system.

Do not replace the existing pipeline. Extend it while preserving the current working implementation.

---

# 1. Experimental Structure

Run four independent experiments.

### Experiment 1: Easy

Training and testing should contain:

```text
Genuine
Altered-Easy
```

### Experiment 2: Medium

Training and testing should contain:

```text
Genuine
Altered-Medium
```

### Experiment 3: Hard

Training and testing should contain:

```text
Genuine
Altered-Hard
```

### Experiment 4: Combined

Training and testing should contain:

```text
Genuine
Altered-Easy
Altered-Medium
Altered-Hard
```

The combined experiment must treat all altered fingerprints as the Fake class.

---

# 2. Important Data Leakage Requirement

Do not extract or select samples in a way that allows the same fingerprint identity to appear in both training and testing.

The train/test split must happen before any operation that learns information from the data.

Any preprocessing parameters that are learned from the training data must be fitted only on training data.

The test set must remain completely unseen until final evaluation.

---

# 3. Labels

Use:

```python
FAKE = 0
GENUINE = 1
```

For Easy:

```text
Genuine → 1
Altered-Easy → 0
```

For Medium:

```text
Genuine → 1
Altered-Medium → 0
```

For Hard:

```text
Genuine → 1
Altered-Hard → 0
```

For Combined:

```text
Genuine → 1
Altered-Easy → 0
Altered-Medium → 0
Altered-Hard → 0
```

---

# 4. LBP Configuration

Use the corrected uniform LBP configuration:

```python
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
LBP_BLOCKS = (8, 8)
LBP_BINS = 10
```

Expected feature dimension:

```text
8 × 8 × 10 = 640
```

Every fingerprint image must therefore produce:

```python
feature_vector.shape == (640,)
```

---

# 5. Create a Reusable Experiment Function

Refactor the current pipeline so that an experiment can be executed using a function similar to:

```python
def run_soco_fing_experiment(
    experiment_name,
    genuine_images,
    fake_images,
    output_dir
):
    ...
```

The function must:

1. Receive genuine images.
2. Receive fake images.
3. Assign the correct labels.
4. Split the data into training and testing sets.
5. Balance the training set only.
6. Extract LBP features.
7. Train the SVM.
8. Evaluate the SVM.
9. Train K-nearest neighbor.
10. Evaluate K-nearest neighbor.
11. Save metrics.
12. Save confusion matrices.
13. Save ROC curves.
14. Save precision-recall curves.
15. Return a structured result dictionary.

---

# 6. Training Balance

The current experiment uses:

```text
240 Genuine
240 Fake
```

for training.

Continue balancing the training set, but perform this independently for every experiment.

For example:

```text
Easy:
Genuine training = 240
Fake training    = 240

Medium:
Genuine training = 240
Fake training    = 240

Hard:
Genuine training = 240
Fake training    = 240
```

For Combined:

```text
Genuine training = 240
Fake training    = 240
```

The 240 fake samples should be sampled from the combined altered categories without allowing one category to completely dominate the experiment.

Use a fixed random seed for reproducibility.

For example:

```python
RANDOM_STATE = 42
```

---

# 7. Do NOT Balance the Test Set

The test set must represent the actual distribution of the available test data.

Do not undersample the test set simply to improve accuracy.

Report the actual:

```text
Genuine test samples
Fake test samples
Total test samples
```

This is important because Presentation Attack Detection should also be evaluated under realistic class distributions.

---

# 8. SVM

Keep the existing SVM GridSearchCV implementation.

The search should evaluate:

```python
param_grid = {
    "svm__C": [...],
    "svm__gamma": [...],
    "svm__kernel": ["rbf"],
    "svm__class_weight": ["balanced"]
}
```

Use stratified cross-validation.

For example:

```python
GridSearchCV(
    pipeline,
    param_grid,
    scoring="roc_auc",
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    n_jobs=-1,
    verbose=1
)
```

Record:

```text
Best parameters
Best cross-validation ROC-AUC
Test ROC-AUC
```

---

# 9. K-nearest Neighbor

Keep the existing K-nearest neighbor GridSearchCV implementation.

Search at least:

```python
param_grid = {
    "knn__n_neighbors": [...],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean"]
}
```

Use the same stratified five-fold cross-validation methodology.

Record:

```text
Best parameters
Best cross-validation ROC-AUC
Test ROC-AUC
```

---

# 10. Required Evaluation Metrics

For every model and every difficulty level, calculate:

### Classification metrics

```text
Accuracy
Balanced Accuracy
Precision
Recall
F1-score
```

Report metrics separately for:

```text
Fake
Genuine
```

Also report:

```text
Macro Precision
Macro Recall
Macro F1
Weighted Precision
Weighted Recall
Weighted F1
```

### PAD-specific metrics

Also calculate:

```text
ROC-AUC
Precision-Recall AUC
Matthews Correlation Coefficient
```

Because the test sets may be imbalanced, do not rely on accuracy alone.

---

# 11. Presentation Attack Detection Error Analysis

Because this is a Presentation Attack Detection experiment, explicitly report:

```text
False Acceptance Rate
False Rejection Rate
```

Using:

```text
False Acceptance:
Fake fingerprint predicted as Genuine

False Rejection:
Genuine fingerprint predicted as Fake
```

Based on the confusion matrix:

```text
                 Predicted
              Fake   Genuine

Actual Fake     TN       FP

Actual Genuine  FN       TP
```

Where:

```text
False Acceptance Rate = FP / (FP + TN)

False Rejection Rate = FN / (FN + TP)
```

Use the actual class meaning rather than blindly using scikit-learn's generic terminology.

Document exactly how these metrics are defined.

---

# 12. Confusion Matrices

Save one confusion matrix for each model and each experiment.

Required files:

```text
results/
├── Easy/
│   ├── SVM_confusion_matrix.png
│   └── KNN_confusion_matrix.png
│
├── Medium/
│   ├── SVM_confusion_matrix.png
│   └── KNN_confusion_matrix.png
│
├── Hard/
│   ├── SVM_confusion_matrix.png
│   └── KNN_confusion_matrix.png
│
└── Combined/
    ├── SVM_confusion_matrix.png
    └── KNN_confusion_matrix.png
```

Label the axes clearly:

```text
Predicted Class
Actual Class
```

with:

```text
Fake
Genuine
```

---

# 13. ROC Curves

Generate ROC curves for every experiment.

Required:

```text
Easy SVM
Easy KNN

Medium SVM
Medium KNN

Hard SVM
Hard KNN

Combined SVM
Combined KNN
```

Also generate one comparison plot containing:

```text
Easy SVM
Medium SVM
Hard SVM
Combined SVM
```

and another containing:

```text
Easy KNN
Medium KNN
Hard KNN
Combined KNN
```

Include the ROC-AUC in the legend.

---

# 14. Precision-Recall Curves

Because the test distributions can be highly imbalanced, generate Precision-Recall curves as well.

Required:

```text
Easy SVM
Easy KNN

Medium SVM
Medium KNN

Hard SVM
Hard KNN

Combined SVM
Combined KNN
```

Save them under:

```text
results/
```

---

# 15. Results CSV

Create a single CSV:

```text
results/SOCOFing_LBP_all_results.csv
```

Each row should represent one model on one difficulty level.

Expected structure:

```text
Experiment,Model,TrainSamples,TestSamples,
TrainGenuine,TrainFake,
TestGenuine,TestFake,
FeatureDimension,
BestParams,
CV_ROC_AUC,
ROC_AUC,
PR_AUC,
Accuracy,
BalancedAccuracy,
FakePrecision,
FakeRecall,
FakeF1,
GenuinePrecision,
GenuineRecall,
GenuineF1,
MacroF1,
MCC,
FalseAcceptanceRate,
FalseRejectionRate
```

There should be at least:

```text
Easy       SVM
Easy       KNN
Medium     SVM
Medium     KNN
Hard       SVM
Hard       KNN
Combined   SVM
Combined   KNN
```

Therefore:

```text
8 rows minimum
```

---

# 16. Console Output

The console output must clearly separate the four experiments.

Use:

```text
============================================================
SOCOFing LBP PAD EXPERIMENT: EASY
============================================================

Genuine samples:
Altered-Easy samples:

Training:
Genuine:
Fake:

Testing:
Genuine:
Fake:

Feature dimension: 640

--- SVM ---
Best parameters:
CV ROC-AUC:
Test ROC-AUC:
PR-AUC:
Accuracy:
Balanced Accuracy:
Macro F1:
MCC:
False Acceptance Rate:
False Rejection Rate:

--- KNN ---
...
```

Repeat for:

```text
MEDIUM
HARD
COMBINED
```

---

# 17. LBP Verification

Before running the complete experiments, demonstrate that LBP is actually being extracted.

For one fingerprint from each category:

```text
Genuine
Altered-Easy
Altered-Medium
Altered-Hard
```

save:

```text
results/lbp_examples/
```

with:

```text
genuine_original.png
genuine_lbp.png

easy_original.png
easy_lbp.png

medium_original.png
medium_lbp.png

hard_original.png
hard_lbp.png
```

Also save a CSV containing an example feature vector:

```text
results/lbp_examples/sample_lbp_features.csv
```

It should contain:

```text
image
label
feature_001
feature_002
...
feature_640
```

Verify:

```python
assert feature_vector.shape == (640,)
```

---

# 18. Feature Matrix Verification

Before training each model, print:

```text
X_train shape: (N, 640)
X_test shape: (M, 640)
y_train shape: (N,)
y_test shape: (M,)
```

The actual values must come from the experiment.

Do not hard-code the values.

Also verify:

```python
assert X_train.shape[1] == 640
assert X_test.shape[1] == 640
assert not np.isnan(X_train).any()
assert not np.isnan(X_test).any()
assert not np.isinf(X_train).any()
assert not np.isinf(X_test).any()
```

---

# 19. Dataset Breakdown

Before running the experiments, print:

```text
============================================================
SOCOFing DATASET BREAKDOWN
============================================================

Genuine:
Altered-Easy:
Altered-Medium:
Altered-Hard:
Total altered:

============================================================
EXPERIMENT DATASETS
============================================================

Easy:
Genuine:
Fake:

Medium:
Genuine:
Fake:

Hard:
Genuine:
Fake:

Combined:
Genuine:
Fake:
```

Use actual values.

---

# 20. Final Comparison Table

At the end of the program, print a table similar to:

```text
====================================================================
SOCOFing LBP PAD FINAL COMPARISON
====================================================================

Experiment    Model    ROC-AUC    PR-AUC    Accuracy    Balanced Acc.    Macro F1
Easy          SVM
Easy          KNN
Medium        SVM
Medium        KNN
Hard          SVM
Hard          KNN
Combined      SVM
Combined      KNN
====================================================================
```

Also print:

```text
Best SVM experiment:
Best KNN experiment:
Best overall experiment:
```

Determine these values programmatically from ROC-AUC.

---

# 21. Difficulty Analysis

After all experiments have completed, calculate and print the change in performance from Easy to Medium to Hard.

For example:

```text
SVM ROC-AUC:

Easy:
Medium:
Hard:
Combined:

KNN ROC-AUC:

Easy:
Medium:
Hard:
Combined:
```

Then calculate:

```text
Easy → Medium change
Medium → Hard change
Easy → Hard change
```

This will allow the research discussion to determine whether increasing fingerprint alteration difficulty causes performance degradation.

Do not write a conclusion based on expected behaviour. The conclusion must be based on the actual experimental results.

---

# 22. Reproducibility

Use:

```python
RANDOM_STATE = 42
```

where applicable.

The experiment must produce reproducible data splits and sampling.

Save the final configuration used for the experiment:

```text
results/experiment_config.json
```

It should contain:

```json
{
    "lbp_points": 8,
    "lbp_radius": 1,
    "lbp_method": "uniform",
    "lbp_blocks": [8, 8],
    "lbp_bins": 10,
    "feature_dimension": 640,
    "random_state": 42,
    "cv_folds": 5
}
```

---

# 23. Important Interpretation Requirement

Do not assume that:

```text
Easy > Medium > Hard
```

will necessarily occur.

The experiment must determine this empirically.

If Hard performs better than Easy, report it and investigate possible reasons.

Likewise, do not select the model based solely on accuracy.

For this PAD problem, ROC-AUC, PR-AUC, balanced accuracy, false acceptance rate, false rejection rate, and class-specific recall are important.

---

# 24. Proposed Code Architecture

Prefer the following structure:

```text
project/
│
├── config.py
│
├── features/
│   └── lbp.py
│
├── data/
│   └── socofing_loader.py
│
├── models/
│   ├── svm.py
│   └── knn.py
│
├── evaluation/
│   └── metrics.py
│
├── experiments/
│   └── socofing_experiment.py
│
├── results/
│   ├── Easy/
│   ├── Medium/
│   ├── Hard/
│   ├── Combined/
│   └── lbp_examples/
│
└── run_soco_fing.py
```

Do not create unnecessary duplicate implementations. Reuse the existing feature extraction, model training and evaluation functions where possible.

---

# 25. Final Acceptance Criteria

The implementation is complete only when:

- Easy experiment runs successfully.
- Medium experiment runs successfully.
- Hard experiment runs successfully.
- Combined experiment runs successfully.
- Both SVM and K-nearest neighbor are evaluated for every experiment.
- Every image produces a 640-dimensional LBP feature vector.
- Training data is balanced independently for each experiment.
- Test data remains naturally distributed.
- No data leakage occurs.
- ROC-AUC is reported.
- Precision-recall AUC is reported.
- Balanced accuracy is reported.
- Macro F1 is reported.
- Matthews correlation coefficient is reported.
- False Acceptance Rate is reported.
- False Rejection Rate is reported.
- Confusion matrices are saved.
- ROC curves are saved.
- Precision-recall curves are saved.
- LBP example images are saved.
- Example 640-dimensional feature vectors are saved.
- All results are saved to one CSV.
- A final comparison table is printed.
- Easy, Medium, Hard and Combined performance can be directly compared.
- The actual experimental results, rather than assumptions, are used for interpretation.

Do not fabricate or estimate any result. Every reported number must come directly from the executed experiment.