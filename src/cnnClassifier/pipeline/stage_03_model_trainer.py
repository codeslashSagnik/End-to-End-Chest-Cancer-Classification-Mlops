from pathlib import Path
import sys
import os

root_directory = Path('e:/cv projects/Deep Learning MLFlow and DVC')
sys.path.append(str(root_directory / 'src'))

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training,TrainingConfig
from cnnClassifier import logger

STAGE_NAME = "Training"



class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = TrainingConfig()
        training = Training(config=config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    #python stage_03_model_trainer.py