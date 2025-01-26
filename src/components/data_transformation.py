import sys
import os
from dataclasses import dataclass
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src import utils
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

    

class DataTranformation:
    
    def __init__(self):
        pass        
        
    def get_data_transformer_obj(self,numerical_columns,categorical_columns):
        
        try:
           
            logging.info('Data transformation pipeline initiated')
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path, preprocessor_obj_file_path):
        
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Train and test read into df')
            
            preprocessor = self.get_data_transformer_obj(numerical_columns=numerical_columns, categorical_columns=categorical_columns)
            
            # print(f'preprossor {preprocessor}')
            # print(f'train_df {train_df.head(2)}')
            # print(f'test_df {test_df.head(2)}')
            
            target_column_name = "math_score"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info('Preprocessing completed')
            
            utils.save_object(
                file_path = preprocessor_obj_file_path,
                obj = preprocessor
            )
            
            logging.info('Pickle file created for the data transformer')
            
            return (
                train_arr,
                test_arr,
                preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
    