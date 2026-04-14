import numpy as np
import pandas as pd
import optuna
import joblib
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

class FusionRiskModel:
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.model = None
        self.feature_names = None
        self.metadata = {}

    def knn_fusion(self, retinal_df, cvd_df, k=5):
        """
        KNN-based quartile matching (Page 6).
        Finds 5 nearest CVD records for each retinal sample to create a 
        fused multimodal feature set.
        """
        # We match based on shared demographic/risk proxies (age, sex, bp_meds) 
        # or matching vessel-derived CVD scores to patient profiles.
        match_cols = ['age', 'sex', 'sysBP', 'diaBP'] 
        
        # Ensure both dataframes have the matching columns
        # In a real pipeline, these would be extracted from metadata or user input
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(cvd_df[match_cols])
        
        distances, indices = nn.kneighbors(retinal_df[match_cols])
        
        fused_data = []
        for i, idx_list in enumerate(indices):
            # Aggregate the 5 nearest CVD neighbors (mean)
            cvd_profile = cvd_df.iloc[idx_list].mean()
            # Combine with retinal features
            combined = pd.concat([retinal_df.iloc[i], cvd_profile])
            fused_data.append(combined)
            
        return pd.DataFrame(fused_data)

    def optimize_hyperparameters(self, X, y):
        """Optuna hyperparameter tuning (50 trials, 5-fold CV)."""
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            clf = XGBClassifier(**param)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Use AUC for optimization
            score = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def train(self, X, y):
        """Full training pipeline with SMOTE, Tuning, and Calibration."""
        self.feature_names = X.columns.tolist()
        
        # 1. Stratified Split
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(skf.split(X, y))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 2. SMOTE Oversampling (Training Fold ONLY - Page 6)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # 3. Hyperparameter Tuning
        best_params = self.optimize_hyperparameters(X_resampled, y_resampled)
        
        # 4. Train Calibrated Classifier (Page 7)
        base_model = XGBClassifier(**best_params)
        self.model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
        self.model.fit(X_resampled, y_resampled)

        # 5. Metadata for Save
        probs = self.model.predict_proba(X_test)[:, 1]
        self.metadata = {
            'threshold': 0.5, # Default, can be tuned via Youden's J
            'auc': roc_auc_score(y_test, probs),
            'params': best_params,
            'feature_names': self.feature_names
        }
        
        print(f"Model Trained. Test AUC: {self.metadata['auc']:.4f}")

    def save_model(self, path="fusion_risk_model.joblib"):
        """Save model with full metadata (Page 7)."""
        data = {
            'model': self.model,
            'metadata': self.metadata
        }
        joblib.dump(data, path)
        print(f"Model saved to {path}")

    def predict_risk(self, X):
        """Returns true probability output."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X)[:, 1]