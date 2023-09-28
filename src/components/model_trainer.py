import os
import sys
import joblib
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,model_evaluation,get_best_model
from src.components import data_transformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting training and test input data")
            X_train,X_test,y_train,y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            models={
                "decision_tree":DecisionTreeRegressor(),
                "random_forest":RandomForestRegressor(),
                "gradient_boosting":GradientBoostingRegressor(),
                "linear_regression":LinearRegression(),
                "xgb_regressor":XGBRegressor(),
                # "k-neighbour_classifier":KNeighborsRegressor(),
                "catboosting_classifier":CatBoostRegressor(verbose=True),
                "adaboost_classifier":AdaBoostRegressor(),
            }
            params={
                "decision_tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "random_forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "gradient_boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "linear_regression":{},
                "xgb_regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "catboosting_classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "adaboost_classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            model_report=model_evaluation(X_train,y_train,X_test,y_test,models,parameters=params)
            best_model_info=get_best_model(model_report)
            model = best_model_info['model']
            model_r2_score=best_model_info['r2_score']
            model.set_params(**best_model_info['parameters'])
            model.fit(X_train, y_train)

            logging.info("Model performing best on train and test set found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
                type=joblib
            )
            # print(model_r2_score)
        except Exception as e:
            raise CustomException(e,sys)
     

if __name__=='__main__':
    trans=data_transformation.DataTransformation()
    a, b = trans.initiate_data_transformation('G:\\Users\\USER\\Desktop\\AdultCensus_\\mlProject\\artifacts\\train.csv','G:\\Users\\USER\\Desktop\\AdultCensus_\\mlProject\\artifacts\\test.csv')
    train_=ModelTrainer()
    train_.initiate_model_trainer(a, b)