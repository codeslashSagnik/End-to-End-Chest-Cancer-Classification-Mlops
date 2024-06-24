import os
import zipfile
import gdown
import sys
import sys
from pathlib import Path
root_directory = Path('e:/cv projects/Deep Learning MLFlow and DVC')
sys.path.append(str(root_directory / 'src'))
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
import shutil

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            
    def copy_data_to_main_directory(self):
        """
        Copies the data from the artifacts directory to the main data directory
        """
        try:
            source_dir = 'e:/cv projects/Deep Learning MLFlow and DVC/src/cnnClassifier/pipeline/artifacts/data_ingestion/Data/'
            dest_dir = os.path.join(root_directory, 'data', 'Data')
            os.makedirs(dest_dir, exist_ok=True)
            
            subdirs = ['train', 'test', 'valid']
            for subdir in subdirs:
                src_path = os.path.join(source_dir, subdir)
                dest_path = os.path.join(dest_dir, subdir)
                if os.path.exists(src_path):
                    logger.info(f"Copying from {src_path} to {dest_path}")
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                    logger.info(f"Copied {src_path} to {dest_path}")
                else:
                    logger.warning(f"Source path {src_path} does not exist")
        except Exception as e:
            logger.exception("Error copying data to main directory", exc_info=True)
            raise e
