import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exception import custom_exception
from src.logger import logging
from dataclasses import dataclass
from sklearn.utils import resample
from PIL import Image
import albumentations as A
import cv2

@dataclass
class data_transform_config:
    client1_data_path: str = os.path.join('artifacts', 'data', 'Client1')
    client2_data_path: str = os.path.join('artifacts', 'data', 'Client2')
    client3_data_path: str = os.path.join('artifacts', 'data', 'Client3')  # Added Client3 path
    preprocessor_ob_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class data_transformation:
    def __init__(self):
        self.transform_config = data_transform_config()
        self.augmentation = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.GaussNoise(),
                A.GaussianBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.OpticalDistortion(p=0.2),
                A.PiecewiseAffine(p=0.2),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.HueSaturationValue(p=0.5),
        ])
    
    def balance_dataset(self, client_paths):
        for client_path in client_paths:
            categories = os.listdir(client_path)  # Dynamically get category names from folder names
            for category in categories:
                category_path = os.path.join(client_path, category)
                if not os.path.exists(category_path):
                    continue  # Skip if category folder does not exist

                images = os.listdir(category_path)
                target_size = 600  # Set target size to 600
                
                # Downsample if greater than target size
                if len(images) > target_size:
                    downsampled = resample(images, n_samples=target_size, random_state=42)
                    for img in set(images) - set(downsampled):
                        os.remove(os.path.join(category_path, img))
                
                # Upsample if less than target size
                elif len(images) < target_size:
                    num_to_generate = target_size - len(images)
                    for i in range(num_to_generate):
                        source_img = np.random.choice(images)
                        img_path = os.path.join(category_path, source_img)
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        augmented = self.augmentation(image=image)['image']
                        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                        
                        new_img_path = os.path.join(category_path, f"augmented_{i}_{source_img}")
                        cv2.imwrite(new_img_path, augmented)

    def initiate_data_transformation(self):
        try:
            logging.info("Starting data transformation")
            
            client_paths = [
                self.transform_config.client1_data_path,
                self.transform_config.client2_data_path,
                self.transform_config.client3_data_path  # Include Client3 path
            ]
            self.balance_dataset(client_paths)  # No need to pass categories now
            
            logging.info("Data transformation completed successfully")
            return (self.transform_config.client1_data_path,
                    self.transform_config.client2_data_path,
                    self.transform_config.client3_data_path,  # Return Client3 path
                    self.transform_config.preprocessor_ob_file_path)
        
        except Exception as e:
            raise custom_exception(e, sys)      

def main():
    try:
        logging.info("Starting data transformation process")
        dt = data_transformation()
        client1_path, client2_path, client3_path, preprocessor_path = dt.initiate_data_transformation()
        logging.info(f"Data transformation completed. Transformed data saved at:")
        logging.info(f"Client 1: {client1_path}")
        logging.info(f"Client 2: {client2_path}")
        logging.info(f"Client 3: {client3_path}")  # Log Client3 path
        logging.info(f"Preprocessor object saved at: {preprocessor_path}")
    except Exception as e:
        logging.error("An error occurred during data transformation")
        raise custom_exception(e, sys)

if __name__ == "__main__":
    main()