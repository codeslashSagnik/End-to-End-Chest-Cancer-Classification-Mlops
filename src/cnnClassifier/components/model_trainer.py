import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
import sys
import os

root_directory = Path('e:/cv projects/Deep Learning MLFlow and DVC')
sys.path.append(str(root_directory / 'src'))
#from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path



STAGE_NAME = "Training"

class TrainingConfig:
    def __init__(self):
        self.params_image_size = [224, 224, 3]  
        self.params_batch_size = 16  
        self.params_epochs = 1  
        self.params_is_augmentation = True
        self.updated_base_model_path = r"e:/cv projects/Deep Learning MLFlow and DVC/src/cnnClassifier/pipeline/artifacts/prepare_base_model/updated_base_model.h5"
        self.trained_model_path = r"e:/cv projects/Deep Learning MLFlow and DVC/src/cnnClassifier/pipeline/artifacts/training/model.h5"
        self.training_data_dir = r"e:/cv projects/Deep Learning MLFlow and DVC/data/Data"  # Raw string for directory

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        print(f"Loading base model from: {self.config.updated_base_model_path}")
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        
        # Assuming the last layer before the output layer is self.model.layers[-2]
        x = self.model.layers[-2].output  # Output of the layer before the Dense output layer
        
        num_classes = 3  # Number of classes in your dataset
        predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs=self.model.input, outputs=predictions)
        
        # Compile the model to apply the changes
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Debug: Print model summary and output shape after adjustment
        self.model.summary()
        print("Adjusted Model Output Shape:", self.model.output_shape)

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        print(f"Using training data directory: {self.config.training_data_dir}")
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data_dir,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        print("Train Generator Labels Shape:", self.train_generator.labels.shape)
        print("Valid Generator Labels Shape:", self.valid_generator.labels.shape)

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )