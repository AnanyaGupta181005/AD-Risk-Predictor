import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Add this if missing
from sklearn.calibration import calibration_curve # Ensure this is from .calibration

class InterpretabilityEngine:
    def __init__(self, model, feature_names):
        # For CalibratedClassifierCV, we reach into the base XGBoost model
        self.model = model.base_estimator if hasattr(model, 'base_estimator') else model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)

    def generate_calibration_plot(self, y_true, y_probs, save_path="calibration_plot.png"):
        """Reliability diagram for risk model."""
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Multimodal Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def get_top_risk_explanations(self, X_test, y_probs, top_n=3):
        """Extract SHAP force tables for top 3 highest-risk predictions (Page 7)."""
        top_indices = np.argsort(y_probs)[-top_n:][::-1]
        X_top = X_test.iloc[top_indices]
        
        shap_values = self.explainer.shap_values(X_top)
        
        explanations = []
        for i in range(top_n):
            # Create a simple table of top 5 contributing features for this patient
            patient_shap = pd.Series(shap_values[i], index=self.feature_names)
            top_contributors = patient_shap.sort_values(key=abs, ascending=False).head(5)
            
            explanations.append({
                "patient_idx": top_indices[i],
                "risk_score": y_probs[top_indices[i]],
                "top_features": top_contributors.to_dict()
            })
            
        return explanations