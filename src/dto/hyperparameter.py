from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class DecisionTreeParams:
    criterion: List[str]
    splitter: List[str]
    max_features: List[str]

@dataclass
class RandomForestParams:
    criterion: List[str]
    max_features: List[Optional[str]]  # null becomes Optional
    n_estimators: List[int]

@dataclass
class GradientBoostingParams:
    loss: List[str]
    learning_rate: List[float]
    subsample: List[float]
    criterion: List[str]
    max_features: List[str]
    n_estimators: List[int]

@dataclass
class LinearRegressionParams:
    pass  # Empty config

@dataclass
class XGBRegressorParams:
    learning_rate: List[float]
    n_estimators: List[int]

@dataclass
class CatBoostingRegressorParams:
    depth: List[int]
    learning_rate: List[float]
    iterations: List[int]

@dataclass
class AdaBoostRegressorParams:
    learning_rate: List[float]
    loss: List[str]
    n_estimators: List[int]

@dataclass
class HyperparametersConfig:
    decision_tree: Dict[str, Any]
    random_forest: Dict[str, Any]
    gradient_boosting: Dict[str, Any]
    linear_regression: Dict[str, Any]
    xgb_regressor: Dict[str, Any]
    cat_boosting_regressor: Dict[str, Any]
    ada_boost_regressor: Dict[str, Any]