"""
Tabular Preprocessor and Feature Engineering Module.

Transforms raw creator video metadata into model-ready numerical matrices:
- Text heuristic engineering (title length, uppercase ratio, hook punctuation, viral keywords)
- Temporal feature extraction (hour, day-of-week, weekend indicator, cyclical sin/cos encodings)
- Channel baseline log-scaling
- Fitted standardization via StandardScaler
- Serialization and single-record inference support for FastAPI serving
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class VideoPreprocessor:
    """Production tabular preprocessor for creator video performance prediction."""

    VIRAL_KEYWORDS: List[str] = [
        "SECRET", "INSANE", "UNBELIEVABLE", "TRUTH", "WARNING", "MISTAKE",
        "STOP", "SHOCKING", "NEVER", "OFFICIAL", "REVEALED", "PROOF",
        "BEST", "WORST", "EXPOSED", "$", "50 HOURS", "ULTIMATE"
    ]

    NUMERICAL_FEATURES: List[str] = [
        "duration_seconds",
        "tags_count",
        "title_char_length",
        "title_word_count",
        "title_uppercase_ratio",
        "published_hour",
        "published_day_of_week",
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
        "log_channel_median_views",
        "viral_keywords_count",
    ]

    BOOLEAN_FEATURES: List[str] = [
        "has_question_or_exclamation",
        "is_weekend",
        "is_short",
    ]

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        boolean_features: Optional[List[str]] = None,
    ) -> None:
        """Initialize the preprocessor with specified feature definitions.
        
        Args:
            numerical_features: List of numerical column names to standardize.
            boolean_features: List of binary/boolean column names (kept in {0, 1}).
        """
        self.numerical_features = list(numerical_features or self.NUMERICAL_FEATURES)
        self.boolean_features = list(boolean_features or self.BOOLEAN_FEATURES)
        self.scaler = StandardScaler()
        self.is_fitted: bool = False
        self._feature_names: List[str] = self.numerical_features + self.boolean_features
        self._imputation_values: Dict[str, float] = {}

    @property
    def feature_names(self) -> List[str]:
        """Returns ordered list of all engineered feature names."""
        return list(self._feature_names)

    @staticmethod
    def _parse_datetime(dt_val: Any) -> datetime:
        """Safely parse various datetime representations into a datetime object."""
        if isinstance(dt_val, datetime):
            return dt_val
        if isinstance(dt_val, str):
            # Clean possible whitespace
            dt_str = dt_val.strip()
            # Try ISO format
            try:
                return datetime.fromisoformat(dt_str)
            except ValueError:
                pass
            # Try common format variants
            for fmt in (
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    return datetime.strptime(dt_str, fmt)
                except ValueError:
                    continue
        # Fallback to current UTC time if unparseable
        return datetime.now()

    def engineer_features(self, df_or_records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Extract engineered tabular features from raw video inputs.
        
        Args:
            df_or_records: Raw pandas DataFrame or list of dicts.
            
        Returns:
            DataFrame containing engineered numerical and boolean features.
        """
        if isinstance(df_or_records, list):
            df = pd.DataFrame(df_or_records)
        else:
            df = df_or_records.copy()

        # 1. Title Heuristics
        titles = df["video_title"].fillna("").astype(str)
        df["title_char_length"] = titles.str.len()
        df["title_word_count"] = titles.apply(lambda t: len(t.split()))
        
        def compute_uppercase_ratio(title: str) -> float:
            letters = [c for c in title if c.isalpha()]
            if not letters:
                return 0.0
            return sum(1 for c in letters if c.isupper()) / len(letters)

        df["title_uppercase_ratio"] = titles.apply(compute_uppercase_ratio)
        df["has_question_or_exclamation"] = titles.apply(
            lambda t: int("?" in t or "!" in t)
        )
        
        def count_viral_keywords(title: str) -> int:
            t_upper = title.upper()
            return sum(1 for kw in self.VIRAL_KEYWORDS if kw in t_upper)

        df["viral_keywords_count"] = titles.apply(count_viral_keywords)

        # 2. Temporal Heuristics
        parsed_dts = df["published_at"].apply(self._parse_datetime)
        df["published_hour"] = parsed_dts.apply(lambda dt: dt.hour)
        df["published_day_of_week"] = parsed_dts.apply(lambda dt: dt.weekday())
        df["is_weekend"] = df["published_day_of_week"].apply(lambda d: int(d in (5, 6)))

        # Cyclical Encodings (hour: period 24, day: period 7)
        two_pi = 2.0 * math.pi
        df["sin_hour"] = df["published_hour"].apply(lambda h: math.sin(two_pi * h / 24.0))
        df["cos_hour"] = df["published_hour"].apply(lambda h: math.cos(two_pi * h / 24.0))
        df["sin_day"] = df["published_day_of_week"].apply(lambda d: math.sin(two_pi * d / 7.0))
        df["cos_day"] = df["published_day_of_week"].apply(lambda d: math.cos(two_pi * d / 7.0))

        # 3. Video Characteristics
        df["duration_seconds"] = pd.to_numeric(df.get("duration_seconds", 600.0), errors="coerce").fillna(600.0)
        df["is_short"] = (df["duration_seconds"] <= 60.0).astype(int)

        if "tags_count" in df.columns:
            df["tags_count"] = pd.to_numeric(df["tags_count"], errors="coerce").fillna(0).astype(int)
        elif "tags" in df.columns:
            df["tags_count"] = df["tags"].apply(lambda tg: len(tg) if isinstance(tg, list) else 0)
        else:
            df["tags_count"] = 0

        # 4. Channel Historical Context
        channel_median = pd.to_numeric(df.get("channel_median_views", 10000.0), errors="coerce").fillna(10000.0)
        # Defensive clipping against negative or zero values
        channel_median = channel_median.clip(lower=1.0)
        df["log_channel_median_views"] = np.log1p(channel_median)

        # Ensure all required features are present
        for col in self.numerical_features:
            if col not in df.columns:
                df[col] = 0.0
        for col in self.boolean_features:
            if col not in df.columns:
                df[col] = 0

        return df

    def fit(self, df_or_records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> "VideoPreprocessor":
        """Compute feature engineering, calculate imputation medians, and fit StandardScaler.
        
        Args:
            df_or_records: Training dataset.
            
        Returns:
            Fitted VideoPreprocessor instance.
        """
        engineered_df = self.engineer_features(df_or_records)

        # Compute median imputation values on numerical features
        self._imputation_values = {}
        for col in self.numerical_features:
            self._imputation_values[col] = float(engineered_df[col].median(skipna=True) or 0.0)

        # Impute any missing values before fitting scaler
        num_matrix = engineered_df[self.numerical_features].fillna(self._imputation_values).to_numpy(dtype=np.float32)
        self.scaler.fit(num_matrix)
        self.is_fitted = True
        return self

    def transform(
        self,
        df_or_records: Union[pd.DataFrame, List[Dict[str, Any]]],
        as_dataframe: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transform inputs into standardized feature matrix.
        
        Args:
            df_or_records: Dataset to transform.
            as_dataframe: If True, returns a pd.DataFrame with named columns, else np.ndarray.
            
        Returns:
            Standardized feature matrix of shape (N, len(numerical) + len(boolean)).
        """
        if not self.is_fitted:
            raise RuntimeError("VideoPreprocessor must be fitted before calling transform().")

        engineered_df = self.engineer_features(df_or_records)

        # Impute numerical features
        num_df = engineered_df[self.numerical_features].fillna(self._imputation_values)
        num_scaled = self.scaler.transform(num_df.to_numpy(dtype=np.float32))

        # Boolean features kept in {0, 1}
        bool_matrix = engineered_df[self.boolean_features].fillna(0).to_numpy(dtype=np.float32)

        # Horizontally stack: [scaled_numerical, boolean]
        full_matrix = np.hstack([num_scaled, bool_matrix]).astype(np.float32)

        if as_dataframe:
            return pd.DataFrame(full_matrix, columns=self.feature_names)
        return full_matrix

    def fit_transform(
        self,
        df_or_records: Union[pd.DataFrame, List[Dict[str, Any]]],
        as_dataframe: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit to dataset and return transformed feature matrix."""
        return self.fit(df_or_records).transform(df_or_records, as_dataframe=as_dataframe)

    def transform_single(self, record: Dict[str, Any]) -> np.ndarray:
        """Transform a single video record for low-latency API inference.
        
        Args:
            record: Dictionary containing single video observation.
            
        Returns:
            1D numpy array of shape (num_features,).
        """
        transformed = self.transform([record], as_dataframe=False)
        return transformed[0]

    def save(self, file_path: Union[str, Path]) -> Path:
        """Serialize fitted preprocessor to disk using joblib.
        
        Args:
            file_path: Destination path.
            
        Returns:
            Path where file was saved.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "VideoPreprocessor":
        """Load fitted preprocessor from disk.
        
        Args:
            file_path: Path to serialized preprocessor.
            
        Returns:
            Loaded VideoPreprocessor instance.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessor artifact not found at {path.resolve()}")
        preprocessor = joblib.load(path)
        if not isinstance(preprocessor, cls):
            raise TypeError(f"Loaded object is of type {type(preprocessor)}, expected {cls}.")
        return preprocessor
