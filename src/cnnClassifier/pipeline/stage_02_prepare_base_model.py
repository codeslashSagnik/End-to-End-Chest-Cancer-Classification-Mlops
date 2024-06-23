from pathlib import Path
import sys
root_directory = Path('e:/cv projects/Deep Learning MLFlow and DVC')
sys.path.append(str(root_directory / 'src'))
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger



STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_filepath = root_directory / 'config/config.yaml'
        params_filepath = root_directory / 'params.yaml'

        # Debug print to check file paths
        print(f"Config file path: {config_filepath}")
        print(f"Params file path: {params_filepath}")

        if not config_filepath.exists():
            raise FileNotFoundError(f"{config_filepath} does not exist.")
        if not params_filepath.exists():
            raise FileNotFoundError(f"{params_filepath} does not exist.")
        
        config = ConfigurationManager(config_filepath=config_filepath, params_filepath=params_filepath)
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
