import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
import yaml
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i, (model_name, model) in enumerate(models.items()):
            logging.info(f"Evaluating model: {model_name}")

            param_grid = params.get(model_name, {})
            logging.debug(f"Parameter grid: {param_grid}")

            try:
                # Perform Grid Search
                gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
                gs.fit(X_train, y_train)

                # Set the best parameters to the model
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_model_score

                logging.info(f"Model {model_name} trained successfully.")
                logging.debug(f"Train Score: {train_model_score}, Test Score: {test_model_score}")

            except Exception as model_error:
                logging.error(f"Error training model {model_name}: {model_error}")
                continue

        return report

    except Exception as e:
        logging.exception("Exception occurred in evaluate_models")
        raise CustomException(e, sys)
    
    
def load_hyperparameters(filepath):
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params