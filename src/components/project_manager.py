# Central file to manage all the methods and other files
# Code the files according to project goals and data then execute from here

import os
import sys
from src.exception import custom_exception
from src.logger import logging
from data_ingestion import data_ingestion
from data_transformation import data_transformation
from model_trainer import model_trainer

if __name__ == '__main__':
    
    logging.info('Reading the data from source')
    try:
        df = pd.read_csv('#source.csv')        # Add file name here
        logging.info('Data read successfully')
    except Exception as e:
        raise custom_exception(e, sys)
    
    logging.info('Data Ingesion started')
    ingesion_object = data_ingesion()
    train_data, test_data, raw_data = ingesion_object.initiate_data_ingestion(df)
    logging.info('Data Ingestion completed')
    
    logging.info('Data Transformation started')
    transformation_object = data_transformation()
    train_arr,test_arr,preprocessor_path = transformation_object.initiate_data_transoformer()
    logging.info('Data Transformation completed')
    
    logging.info('Model Training started')
    model_trainer = model_trainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path)
    logging.info('Model Training completed')