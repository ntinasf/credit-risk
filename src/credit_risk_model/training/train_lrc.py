from sklearn.linear_model import LogisticRegression
from skopt.space import Categorical, Integer, Real

from credit_risk_model.training.base import BaseModelTrainer


class LRCTrainer(BaseModelTrainer):
    def get_model_key(self) -> str:
        return "lrc"

    def get_estimator(self):
        return LogisticRegression(
            random_state=self.app_config.random_state,
            class_weight="balanced",
        )

    def get_search_space(self) -> dict:
        space = {
            "model__C": Real(0.1, 20, prior="log-uniform"),
            "model__penalty": Categorical(["l2"]),
            "model__solver": Categorical(["liblinear"]),
            "model__max_iter": Integer(1000, 8000),
            "model__class_weight": Categorical(["balanced", None]),
        }

        if self.model_cfg.use_smote:
            space["smote__k_neighbors"] = Integer(3, 10)
            space["smote__sampling_strategy"] = Real(0.5, 1.0, prior="uniform")

        return space
