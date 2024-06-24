from dataclasses import dataclass
from pathlib import Path
import sys
root_directory = Path('e:/cv projects/Deep Learning MLFlow and DVC')
sys.path.append(str(root_directory / 'src'))
config_filepath = root_directory / 'config' / 'config.yaml'
params_filepath = root_directory / 'params.yaml'
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            config = ConfigurationManager(config_filepath, params_filepath)
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            data_ingestion.copy_data_to_main_directory()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    obj = DataIngestionTrainingPipeline()
    obj.main()

#python stage_01_data_ingestion.py