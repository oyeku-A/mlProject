import os
import sys
import dill
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger import logging
from src.utils import save_object
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl') 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        """ This function is resonsible for data transformation """
        try:
            num_features=['reading_score', 'writing_score']
            cat_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("standardize", StandardScaler()),
            ])

            cat_pipeline = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder()),
            ])

            preprocessing = ColumnTransformer([
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features)
            ])
            return preprocessing
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and Test data Read")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

            target_col="math_score"

            input_feature_train_df=train_df.drop(target_col,axis=1)
            target_feature_train_df=train_df[target_col]

            input_feature_test_df=test_df.drop(target_col,axis=1)
            target_feature_test_df=test_df[target_col]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
                type=dill
                )

            return (
                train_arr,
                test_arr,
                # self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)