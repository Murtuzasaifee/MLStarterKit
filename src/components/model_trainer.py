import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, load_hyperparameters
import yaml
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
    
    
class ModelTrainer:
    
    def __init__(self):
        pass

    def initiate_model_trainer(self, train_array, test_array,trained_model_file_path, hyperparameters):
        
        try:
            
            logging.info('Model trainer initiated')
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], 
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            
            models = {
                "decision_tree": DecisionTreeRegressor(),
                "random_forest": RandomForestRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "linear_regression": LinearRegression(),
                "xgb_regressor": XGBRegressor(),
                "cat_boosting_regressor": CatBoostRegressor(verbose=False),
                "ada_boost_regressor": AdaBoostRegressor(),
            }
            
            # hyperparameters=load_hyperparameters('src/config/hyperparameters.yaml')
            print(f"Hyper Parameters are : {hyperparameters}")
            
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train, X_test=X_test,y_test=y_test,models=models,params=hyperparameters)
            
            print(f'Model Report :{model_report}')
            
            ## Get the best model score from the dict
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info('Best model found')
            
            print(f'Best model found is: {best_model_name}')
            
            save_object(trained_model_file_path, best_model)
            
            logging.info('Best model pckl file saved')
            
            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)
            
            return score
            
        except Exception as e:
            raise CustomException(e, sys)
        
    