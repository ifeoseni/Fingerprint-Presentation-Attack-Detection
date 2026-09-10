import os
import sys
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import SOCOFING_DIR, RANDOM_STATE, CV_FOLDS, LBP_POINTS, LBP_RADIUS, LBP_METHOD, LBP_BLOCKS, LBP_BINS, RESULTS_DIR
from socofing_experiment import load_socofing_image_paths, prepare_experiment_data
from models import get_svm_pipeline, get_knn_pipeline
from evaluation_pad import evaluate_pad_model
from lbp_examples import generate_lbp_examples

def run_soco_fing_experiment(experiment_name, genuine_paths, fake_paths, output_csv_list, sample_fraction=1.0):
    print(f"\n{'='*60}")
    print(f"SOCOFing LBP PAD EXPERIMENT: {experiment_name.upper()}")
    print(f"{'='*60}")
    
    print(f"\nGenuine samples: {len(genuine_paths)}")
    print(f"Altered-{experiment_name} samples: {len(fake_paths)}")
    
    X_train, X_test, y_train, y_test = prepare_experiment_data(
        genuine_paths, fake_paths, 
        sample_fraction=sample_fraction, 
        random_state=RANDOM_STATE
    )
    
    train_gen = int((y_train == 1).sum())
    train_fake = int((y_train == 0).sum())
    test_gen = int((y_test == 1).sum())
    test_fake = int((y_test == 0).sum())
    
    print(f"\nTraining:")
    print(f"Genuine: {train_gen}")
    print(f"Fake: {train_fake}")
    
    print(f"\nTesting:")
    print(f"Genuine: {test_gen}")
    print(f"Fake: {test_fake}")
    
    feature_dim = X_train.shape[1]
    print(f"\nFeature dimension: {feature_dim}")
    assert feature_dim == 640, f"Expected 640 dims, got {feature_dim}"
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Run SVM
    svm_grid = get_svm_pipeline(y_train)
    svm_grid.fit(X_train, y_train)
    y_pred_svm = svm_grid.predict(X_test)
    y_prob_svm = svm_grid.predict_proba(X_test)[:, 1] if hasattr(svm_grid, "predict_proba") else svm_grid.decision_function(X_test)
    
    svm_metrics = evaluate_pad_model(
        y_test, y_pred_svm, y_prob_svm,
        model_name="SVM", experiment_name=experiment_name,
        feature_dim=feature_dim, train_genuine=train_gen, train_fake=train_fake,
        best_params=svm_grid.best_params_, cv_roc_auc=svm_grid.best_score_
    )
    output_csv_list.append(svm_metrics)
    
    # Run KNN
    knn_grid = get_knn_pipeline(y_train)
    knn_grid.fit(X_train, y_train)
    y_pred_knn = knn_grid.predict(X_test)
    y_prob_knn = knn_grid.predict_proba(X_test)[:, 1]
    
    knn_metrics = evaluate_pad_model(
        y_test, y_pred_knn, y_prob_knn,
        model_name="KNN", experiment_name=experiment_name,
        feature_dim=feature_dim, train_genuine=train_gen, train_fake=train_fake,
        best_params=knn_grid.best_params_, cv_roc_auc=knn_grid.best_score_
    )
    output_csv_list.append(knn_metrics)
    
    return svm_metrics, knn_metrics

if __name__ == "__main__":
    import numpy as np # needed for assertions in experiment
    
    # Configuration export
    exp_config = {
        "lbp_points": LBP_POINTS,
        "lbp_radius": LBP_RADIUS,
        "lbp_method": LBP_METHOD,
        "lbp_blocks": list(LBP_BLOCKS),
        "lbp_bins": LBP_BINS,
        "feature_dimension": 640,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "experiment_config.json"), "w") as f:
        json.dump(exp_config, f, indent=4)
        
    print("Generating LBP verification examples...")
    generate_lbp_examples(SOCOFING_DIR)

    # ── Dataset Breakdown ──
    real_paths, _ = load_socofing_image_paths(SOCOFING_DIR, "Easy")
    _, easy_paths = load_socofing_image_paths(SOCOFING_DIR, "Easy")
    _, med_paths = load_socofing_image_paths(SOCOFING_DIR, "Medium")
    _, hard_paths = load_socofing_image_paths(SOCOFING_DIR, "Hard")
    _, all_fake_paths = load_socofing_image_paths(SOCOFING_DIR, "All")
    
    print(f"\n{'='*60}")
    print("SOCOFing DATASET BREAKDOWN")
    print(f"{'='*60}")
    print(f"Genuine: {len(real_paths)}")
    print(f"Altered-Easy: {len(easy_paths)}")
    print(f"Altered-Medium: {len(med_paths)}")
    print(f"Altered-Hard: {len(hard_paths)}")
    print(f"Total altered: {len(all_fake_paths)}")
    
    print(f"\n{'='*60}")
    print("EXPERIMENT DATASETS")
    print(f"{'='*60}")
    print("Easy:\nGenuine:", len(real_paths), "\nFake:", len(easy_paths), "\n")
    print("Medium:\nGenuine:", len(real_paths), "\nFake:", len(med_paths), "\n")
    print("Hard:\nGenuine:", len(real_paths), "\nFake:", len(hard_paths), "\n")
    print("Combined:\nGenuine:", len(real_paths), "\nFake:", len(all_fake_paths), "\n")
    
    # Optional sample fraction for fast testing
    SAMPLE_FRACTION = 1 #0.05
    
    csv_results = []
    
    # Run experiments
    svm_easy, knn_easy = run_soco_fing_experiment("Easy", real_paths, easy_paths, csv_results, SAMPLE_FRACTION)
    svm_med, knn_med = run_soco_fing_experiment("Medium", real_paths, med_paths, csv_results, SAMPLE_FRACTION)
    svm_hard, knn_hard = run_soco_fing_experiment("Hard", real_paths, hard_paths, csv_results, SAMPLE_FRACTION)
    svm_comb, knn_comb = run_soco_fing_experiment("Combined", real_paths, all_fake_paths, csv_results, SAMPLE_FRACTION)
    
    # Save CSV
    df = pd.DataFrame(csv_results)
    csv_path = os.path.join(RESULTS_DIR, "SOCOFing_LBP_all_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Final Comparison Table
    print(f"\n{'='*68}")
    print("SOCOFing LBP PAD FINAL COMPARISON")
    print(f"{'='*68}")
    print(f"{'Experiment':<12} {'Model':<6} {'ROC-AUC':<10} {'PR-AUC':<10} {'Accuracy':<10} {'Balanced Acc.':<15} {'Macro F1':<10}")
    for res in csv_results:
        print(f"{res['Experiment']:<12} {res['Model']:<6} {res['ROC_AUC']:<10.4f} {res['PR_AUC']:<10.4f} {res['Accuracy']:<10.4f} {res['BalancedAccuracy']:<15.4f} {res['MacroF1']:<10.4f}")
    print(f"{'='*68}")
    
    # Find best models
    best_svm = max([r for r in csv_results if r['Model'] == 'SVM'], key=lambda x: x['ROC_AUC'])
    best_knn = max([r for r in csv_results if r['Model'] == 'KNN'], key=lambda x: x['ROC_AUC'])
    best_overall = max(csv_results, key=lambda x: x['ROC_AUC'])
    
    print(f"\nBest SVM experiment: {best_svm['Experiment']} ({best_svm['ROC_AUC']:.4f})")
    print(f"Best KNN experiment: {best_knn['Experiment']} ({best_knn['ROC_AUC']:.4f})")
    print(f"Best overall experiment: {best_overall['Experiment']} - {best_overall['Model']} ({best_overall['ROC_AUC']:.4f})")
    
    # Difficulty Analysis
    print("\nSVM ROC-AUC:")
    print(f"Easy: {svm_easy['ROC_AUC']:.4f}")
    print(f"Medium: {svm_med['ROC_AUC']:.4f}")
    print(f"Hard: {svm_hard['ROC_AUC']:.4f}")
    print(f"Combined: {svm_comb['ROC_AUC']:.4f}")
    
    print("\nKNN ROC-AUC:")
    print(f"Easy: {knn_easy['ROC_AUC']:.4f}")
    print(f"Medium: {knn_med['ROC_AUC']:.4f}")
    print(f"Hard: {knn_hard['ROC_AUC']:.4f}")
    print(f"Combined: {knn_comb['ROC_AUC']:.4f}")
    
    print("\nEasy -> Medium change:")
    print(f"SVM: {svm_med['ROC_AUC'] - svm_easy['ROC_AUC']:+.4f}")
    print(f"KNN: {knn_med['ROC_AUC'] - knn_easy['ROC_AUC']:+.4f}")
    
    print("\nMedium -> Hard change:")
    print(f"SVM: {svm_hard['ROC_AUC'] - svm_med['ROC_AUC']:+.4f}")
    print(f"KNN: {knn_hard['ROC_AUC'] - knn_med['ROC_AUC']:+.4f}")
    
    print("\nEasy -> Hard change:")
    print(f"SVM: {svm_hard['ROC_AUC'] - svm_easy['ROC_AUC']:+.4f}")
    print(f"KNN: {knn_hard['ROC_AUC'] - knn_easy['ROC_AUC']:+.4f}")
    print("\nExperiment completed successfully.")
