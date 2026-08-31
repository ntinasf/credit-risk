from skopt.space import Integer, Real

from credit_risk_model.processing.catboost_wrapper import CatBoostSklearnWrapper
from credit_risk_model.processing.preprocessors import build_catboost_pipeline
from credit_risk_model.training.base import BaseModelTrainer


class CatBoostTrainer(BaseModelTrainer):
    def get_model_key(self) -> str:
        return "cat"

    def _build_pipeline(self, estimator):
        """Use CatBoost-specific pipeline with native categorical handling.

        Unlike the generic build_pipeline(), this passes columns through
        without encoding and injects cat_features indices so CatBoost
        can use its internal optimal encoding for categoricals.
        """
        return build_catboost_pipeline(
            model_cfg=self.model_cfg,
            random_state=self.app_config.random_state,
            scale_pos_weight=self._scale_pos_weight(),
        )

    def _scale_pos_weight(self) -> float:
        """Weight on the positive class (1 = good), from config.

        Explicit None check rather than `or`, because a legitimate weight below 1 is
        the whole point here, and 0.0 would silently fall through.
        """
        cfg_value = self.model_cfg.scale_pos_weight
        return 1.0 if cfg_value is None else cfg_value

    def get_estimator(self):
        # scale_pos_weight handles class imbalance (good is the majority), the
        # counterpart to class_weight="balanced" on the other models. Cost is not
        # encoded here. It is applied at the decision threshold.
        # verbose=0 suppresses CatBoost's per-iteration output
        return CatBoostSklearnWrapper(
            scale_pos_weight=self._scale_pos_weight(),
            random_seed=self.app_config.random_state,
            verbose=0,
        )

    def get_search_space(self) -> dict:
        # scale_pos_weight is fixed by the class ratio, not tuned.
        # The search optimises ROC AUC; the operating point is set later.
        return {
            "model__depth": Integer(4, 10),
            "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "model__iterations": Integer(200, 1000),
            "model__l2_leaf_reg": Real(1.0, 10.0, prior="log-uniform"),
            "model__border_count": Integer(32, 255),
        }
