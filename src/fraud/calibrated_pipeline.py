import pandas as pd


class CalibratedLGBMPipeline:
    """LightGBM model + probability calibrator + operating threshold, bundled
    as one picklable object for serving. `calibrator_type` is "platt" (a
    LogisticRegression on the raw score) or "isotonic" (an IsotonicRegression).
    """

    def __init__(self, model, calibrator_type: str, calibrator,
                 selected_features: list, cat_cols: list, categories_map: dict,
                 threshold: float):
        self.model = model
        self.calibrator_type = calibrator_type
        self.calibrator = calibrator
        self.selected_features = selected_features
        self.cat_cols = cat_cols
        self.categories_map = categories_map
        self.threshold = threshold

    def _to_model_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.selected_features].copy()
        for c in self.cat_cols:
            X[c] = pd.Categorical(X[c], categories=self.categories_map[c])
        return X

    def predict_proba_raw(self, df: pd.DataFrame):
        X = self._to_model_matrix(df)
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, df: pd.DataFrame):
        raw = self.predict_proba_raw(df)
        if self.calibrator_type == "platt":
            return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return self.calibrator.predict(raw)

    def predict(self, df: pd.DataFrame):
        return (self.predict_proba(df) >= self.threshold).astype(int)
