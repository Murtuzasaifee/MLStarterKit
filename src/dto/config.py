from dataclasses import dataclass
from src.dto.hyperparameter import HyperparametersConfig

@dataclass
class InputPaths:
    train_data_path: str
    test_data_path: str
    raw_data_path: str
    input_data_path: str
    test_size: float
    random_state: int

@dataclass
class OutputPaths:
    preprocessor_obj_file_path: str
    trained_model_file_path: str

@dataclass
class FilePathsConfig:
    inputs: InputPaths
    outputs: OutputPaths

@dataclass
class MLConfig:
    files: FilePathsConfig
    hyperparameters: HyperparametersConfig  