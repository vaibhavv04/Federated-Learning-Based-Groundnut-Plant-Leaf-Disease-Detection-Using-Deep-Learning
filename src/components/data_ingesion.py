# Handle functions and objects of this file from project_manager

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exception import custom_exception
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# import other required libraries from here

@dataclass
class data_ingesion_config:
    raw_data_path:str=os.path.join('artifacts','data')
    
class data_ingesion:
    def __init__(self):
        self.ingestion_config=data_ingesion_config()

    def initiate_data_ingestion(self, dataset_name):
        logging.info('Data Ingestion methods Starts')
        try:
            os.makedirs(self.ingestion_config.raw_data_path, exist_ok=True)
            logging.info('Raw data path created')
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            # Download dataset from Kaggle
            api.dataset_download_files(dataset_name, path=self.ingestion_config.raw_data_path, unzip=True)
            logging.info(f'Dataset {dataset_name} downloaded from Kaggle')
            
            return self.ingestion_config.raw_data_path
        
        except Exception as e:
            raise custom_exception(e, sys)


if __name__ == "__main__":
    obj=data_ingesion()
    dataset_name = "emmarex/plantdisease"
    dataset_path = obj.initiate_data_ingestion(dataset_name)
    print(dataset_path)
