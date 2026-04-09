from sklearn.ensemble import RandomForestClassifier
from skopt.space import Categorical, Integer, Real

from credit_risk_model.training.base import BaseModelTrainer


class RFTrainer(BaseModelTrainer):
    def get_model_key(self) -> str:
        return "rfc"

    def get_estimator(self):
        # Cost-sensitive class weights match the 5:1 FP/FN cost ratio
        return RandomForestClassifier(
            class_weight=self.model_cfg.class_weight or {0: 1, 1: 5},
            random_state=self.app_config.random_state,
            n_jobs=-1,
        )

    def get_search_space(self) -> dict:
        return {
            "model__n_estimators": Integer(400, 800),
            "model__max_depth": Integer(4, 15),
            "model__min_samples_split": Integer(10, 80),
            "model__min_samples_leaf": Integer(10, 40),
            "model__max_features": Categorical(["sqrt", "log2", 0.3, 0.5]),
            "model__max_samples": Real(0.5, 0.9, prior="uniform"),
            "model__ccp_alpha": Real(0.0, 0.02, prior="uniform"),
            "model__criterion": Categorical(["gini", "entropy"]),
        }
