import os
import sys
import yaml
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTranformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str
    raw_data_path: str
    input_data_path: str
    test_size: float
    random_state: int

class DataIngestion:
    def __init__(self, config_path="src/config/data_ingestion.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return DataIngestionConfig(**config["DataIngestionConfig"])
        except Exception as e:
            raise CustomException(f"Error loading config: {config_path}", sys) from e

    def initiate_data_ingestion(self):
        logging.info("Data ingestion component initiated")
        try:
            df = pd.read_csv(self.config.input_data_path)
            logging.info('Dataset read into a dataframe')

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            df.to_csv(self.config.raw_data_path, index=False, header=True)

            logging.info('Train-test split initiated')
            train_set, test_set = train_test_split(
                df, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state
            )

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed')

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

# Test Code
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTranformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_array, test_array, _)
    print(f'Final score is : {score}')