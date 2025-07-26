import pandas as pd
import joblib
from typing import List

TOP_FEATURES = [
    'bmi', 'income', 'systolic_blood_pressure', 'triglycerides',
    'sedentary_hours_per_day', 'cholesterol', 'exercise_hours_per_week',
    'heart_rate', 'age', 'stress_level',    # бинарный
    'lifestyle_risk', 'active_score',
    'pulse_pressure', 'age_group'
]

class HeartRiskModel:
    def __init__(self, model_path: str = 'model.pkl', threshold: float = 0.5):
        """
        Загружает обученный пайплайн из файла и устанавливает порог для 
        бинаризации вероятностей.
        """
        self.pipeline = joblib.load(model_path)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0,1], got {threshold}")
        self.threshold = threshold

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # создаём новые признаки
        df['lifestyle_risk'] = df['smoking'] + df['alcohol_consumption'] + df['obesity']
        df['active_score']    = df['exercise_hours_per_week'] - df['sedentary_hours_per_day']
        df['pulse_pressure']  = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
        df['age_group'] = pd.qcut(df['age'], 4, labels=False).astype(int)
        return df

    def predict(self, filepath: str) -> List[int]:
        """
        Считает из CSV-признаки, предсказывает вероятности риска и 
        выдаёт список id пациентов с риском >= threshold.
        """
        df = pd.read_csv(filepath)
        df.columns = df.columns \
            .str.lower() \
            .str.replace(' ', '_') \
            .str.replace(':', '') \
            .str.replace('(', '') \
            .str.replace(')', '')
        # фича-инжиниринг
        df = self._feature_engineering(df)
        if 'id' not in df.columns:
            raise KeyError("Input file must contain 'id' column")
        ids = df['id'].tolist()
        X = df[TOP_FEATURES]
        probs = self.pipeline.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        # возвращаем только тех, у кого риск=1
        high_risk_ids = [pid for pid, p in zip(ids, preds) if p == 1]
        return high_risk_ids
