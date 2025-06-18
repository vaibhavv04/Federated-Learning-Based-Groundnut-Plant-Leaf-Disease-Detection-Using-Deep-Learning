import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

# from catboost import CatBoostRegressor
# from sklearn.ensemble  import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor, 
#     RandomForestRegressor
# )
# from sklearn.linear_model import LinearRegression,Lasso,Ridge
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor


@dataclass
class model_trainer_config:
    trained_model_path = os.path.join('artifacts','model.pkl')
    
class model_trainer:
    def __init__(self):
        self.trainer_config = model_trainer_config()
        
    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info("Splitting features and target values")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            
           # Add code to train, evaluate, compare and to select the best model
           # Save the best model
            
        except Exception as e:
            raise CustomException(e,sys)