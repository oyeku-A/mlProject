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
from src.utils import save_object,model_evaluation
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
                "random_forest":RandomForestRegressor(),
                "decision_tree":DecisionTreeRegressor(),
                "gradient_boosting":GradientBoostingRegressor(),
                "linear_regression":LinearRegression(),
                "k-neighbour_classifier":KNeighborsRegressor(),
                "xgb_classifer":XGBRegressor(),
                "catboosting_classifier":CatBoostRegressor(verbose=True),
                "adaboost_classifier":AdaBoostRegressor(),
            }
            model_report=model_evaluation(X_train,y_train,X_test,y_test,models)
            best_model_name=max(model_report,key=model_report.get)
            best_model_score=model_report[best_model_name]
            best_model=f"{0}: {1}".format(best_model_name,best_model_score)

            if best_model_score<0.6:
                raise CustomException("Best model not found")
            logging.info("Model performing best on train and test set found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
                type=joblib
            )

            y_test_pred=models[best_model_name].predict(X_test)
            r2_square=r2_score(y_test,y_test_pred)

            return r2_score
        except Exception as e:
            raise CustomException(e,sys)
     

