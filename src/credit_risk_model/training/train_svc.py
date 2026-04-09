from sklearn.svm import SVC
from skopt.space import Categorical, Integer, Real

from credit_risk_model.training.base import BaseModelTrainer


class SVCTrainer(BaseModelTrainer):
    def get_model_key(self) -> str:
        return "svc"

    def get_estimator(self):
        # probability=True is required for predict_proba() and threshold tuning
        return SVC(
            probability=True,
            random_state=self.app_config.random_state,
        )

    def get_search_space(self) -> dict:
        space = {
            "model__C": Real(1, 15, prior="uniform"),
            "model__gamma": Real(1e-5, 1, prior="log-uniform"),
            "model__kernel": Categorical(["rbf", "linear"]),
            "model__tol": Real(1e-4, 1e-1, prior="log-uniform"),
        }

        if self.model_cfg.use_smote:
            space["smote__k_neighbors"] = Integer(2, 7)
            space["smote__sampling_strategy"] = Real(0.5, 0.9, prior="uniform")
        else:
            space["model__class_weight"] = Categorical(["balanced", None])

        return space
