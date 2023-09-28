import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj, type):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            type.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

    
def model_evaluation(X_train,y_train,X_test,y_test,models,parameters:dict)->dict:
    try:
        report={}
        for model_name,model in models.items():
            parameters_=parameters[model_name]
            grid_search=GridSearchCV(estimator=model,
                                                        param_grid=parameters_,
                                                        cv=3,
                                                        )
            grid_search.fit(X_train,y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train,y_train)
            y_test_pred=model.predict(X_test)
            test_model_score=r2_score(y_test, y_test_pred)
            report[model_name]={
                'model':model,
                'test_model_score':test_model_score,
                'best_parameters':grid_search.best_params_,
            }
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def get_best_model(report):
    try:
        best_model_name = max(report, key=lambda x: report[x]['test_model_score'])
        best_model_r2_score=report[best_model_name]['test_model_score']
        if best_model_r2_score<0.6:
            raise CustomException("Best model less than threshold")
        else:
            best_model=report[best_model_name]['model']
            best_model_parameter=report[best_model_name]['best_parameters']
            logging.info("Best model found")
            return{
                'model_name': best_model_name,
                'r2_score': best_model_r2_score,
                'parameters': best_model_parameter,
                'model': best_model,
                }
    except Exception as e:
        raise CustomException(e,sys)