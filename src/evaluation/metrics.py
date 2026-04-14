import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve # Moved this here
from scipy import stats

class clinicalMetrics:
    @staticmethod
    def get_hd95(mask_true, mask_pred):
        """Hausdorff distance (95th percentile) for segmentation boundary accuracy."""
        # Extract coordinates of edge pixels
        coords_true = np.argwhere(mask_true > 0)
        coords_pred = np.argwhere(mask_pred > 0)
        
        if len(coords_true) == 0 or len(coords_pred) == 0:
            return 100.0 # Penalty for empty masks
            
        d1 = [np.min(np.linalg.norm(coords_pred - p, axis=1)) for p in coords_true]
        d2 = [np.min(np.linalg.norm(coords_true - p, axis=1)) for p in coords_pred]
        
        return np.percentile(d1 + d2, 95)

    @staticmethod
    def bootstrap_auc(y_true, y_probs, n_bootstraps=1000):
        """Bootstrap 95% CI on AUC-ROC (1000 samples)."""
        rng = np.random.RandomState(42)
        bootstrapped_scores = []
        
        for i in range(n_bootstraps):
            indices = rng.randint(0, len(y_probs), len(y_probs))
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = roc_auc_score(y_true[indices], y_probs[indices])
            bootstrapped_scores.append(score)
            
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        
        return np.mean(sorted_scores), (confidence_lower, confidence_upper)

    @staticmethod
    def calculate_nri(y_true, probs_base, probs_new, threshold=0.5):
        """
        Net Reclassification Improvement (NRI) — retinal vs CVD-only.
        Measures if the new model moves patients to the correct risk category.
        """
        y_base = (probs_base >= threshold).astype(int)
        y_new = (probs_new >= threshold).astype(int)
        
        # Event (Alzheimer's) Group
        events = (y_true == 1)
        up_event = np.sum((y_new[events] > y_base[events]))
        down_event = np.sum((y_new[events] < y_base[events]))
        
        # Non-Event Group
        non_events = (y_true == 0)
        up_nonevent = np.sum((y_new[non_events] > y_base[non_events]))
        down_nonevent = np.sum((y_new[non_events] < y_base[non_events]))
        
        n_events = np.sum(events)
        n_nonevents = np.sum(non_events)
        
        nri = ((up_event - down_event) / n_events) - ((up_nonevent - down_nonevent) / n_nonevents)
        return nri

    @staticmethod
    def mcnemar_test(y_true, y_pred1, y_pred2):
        """McNemar's test for error difference significance between models."""
        cm = confusion_matrix(y_pred1 == y_true, y_pred2 == y_true)
        # contingency table: [ [Both correct, M1 correct], [M2 correct, Both wrong] ]
        b = cm[0, 1]
        c = cm[1, 0]
        statistic = (abs(b - c) - 1)**2 / (b + c + 1e-6)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        return statistic, p_value