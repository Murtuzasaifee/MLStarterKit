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


class DataIngestion:
    
    def __init__(self):
        pass
    
    def initiate_data_ingestion(self, train_data_path ,test_data_path, raw_data_path,input_data_path, test_size, random_state):
        logging.info("Data ingestion component initiated")
        try:
            df = pd.read_csv(input_data_path)
            logging.info('Dataset read into a dataframe')

            os.makedirs(os.path.dirname(train_data_path), exist_ok=True)

            df.to_csv(raw_data_path, index=False, header=True)

            logging.info('Train-test split initiated')
            train_set, test_set = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state
            )

            train_set.to_csv(train_data_path, index=False, header=True)
            test_set.to_csv(test_data_path, index=False, header=True)

            logging.info('Data ingestion completed')

            return (
                train_data_path,
                test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)