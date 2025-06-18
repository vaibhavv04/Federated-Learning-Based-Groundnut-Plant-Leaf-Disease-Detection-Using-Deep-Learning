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
    client1_data_path: str = os.path.join('artifacts','data','Client1')
    client2_data_path: str = os.path.join('artifacts','data','Client2')
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
    
    def balance_dataset(self, client1_path, client2_path, categories):
        for category in categories:
            client1_images = os.listdir(os.path.join(client1_path, category))
            client2_images = os.listdir(os.path.join(client2_path, category))
            
            target_size = (len(client1_images) + len(client2_images)) // 2
            
            # Downsample client1
            if len(client1_images) > target_size:
                downsampled = resample(client1_images, n_samples=target_size, random_state=42)
                for img in set(client1_images) - set(downsampled):
                    os.remove(os.path.join(client1_path, category, img))
            
            # Upsample client2 with augmentations
            if len(client2_images) < target_size:
                num_to_generate = target_size - len(client2_images)
                for i in range(num_to_generate):
                    source_img = np.random.choice(client2_images)
                    img_path = os.path.join(client2_path, category, source_img)
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    augmented = self.augmentation(image=image)['image']
                    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                    
                    new_img_path = os.path.join(client2_path, category, f"augmented_{i}_{source_img}")
                    cv2.imwrite(new_img_path, augmented)

    def initiate_data_transformation(self):
        try:
            logging.info("Starting data transformation")
            
            categories = ['Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Bacterial_spot']
            self.balance_dataset(self.transform_config.client1_data_path, 
                                 self.transform_config.client2_data_path, 
                                 categories)
            
            logging.info("Data transformation completed successfully")
            return (self.transform_config.client1_data_path,
                    self.transform_config.client2_data_path,
                    self.transform_config.preprocessor_ob_file_path)
        
        except Exception as e:
            raise custom_exception(e, sys)      
        
def main():
    try:
        logging.info("Starting data transformation process")
        dt = data_transformation()
        client1_path, client2_path, preprocessor_path = dt.initiate_data_transformation()
        logging.info(f"Data transformation completed. Transformed data saved at:")
        logging.info(f"Client 1: {client1_path}")
        logging.info(f"Client 2: {client2_path}")
        logging.info(f"Preprocessor object saved at: {preprocessor_path}")
    except Exception as e:
        logging.error("An error occurred during data transformation")
        raise custom_exception(e, sys)

if __name__ == "__main__":
    main()
    