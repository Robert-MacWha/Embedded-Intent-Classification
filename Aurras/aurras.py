from transformers import DistilBertTokenizerFast
import numpy as np
import os

from .config import Config as config
from .core import data_processing, model

class Aurras():

    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_VARIANT)

    def generate_data_from_file(self):
        
        json_data = data_processing.preprocess_intent_dataset(config.INTENT_SAMPLES, config.DATASET_PATH)
        self.dataset = data_processing.process_triplets(json_data)

        self.dataset = data_processing.tokenize_intent_dataset(self.dataset, self.tokenizer, config.TOKENIZED_PADDING)

        np.save(f'{config.DATASET_PATH}/dataset.npy', self.dataset)


    def load_data_from_file(self):
        if os.path.isfile(f'{config.DATASET_PATH}/dataset.npy'):
            self.dataset = np.load(f'{config.DATASET_PATH}/dataset.npy')
        else:
            print("Couldn't load data from the file - generating a dataset")
            self.generate_data_from_file()

        print(self.dataset.shape)

    def train(self):
        self.__build_model()

        self.model.fit(self.dataset, config.EPOCHS, config.BATCH_SIZE, 1)

    def save(self):
        self.model.save(config.MODEL_PATH)

    def load_model_from_file(self):
        pass

    def get_intent(self, prompt):
        pass

    def __build_model(self):
        self.model = model.Model(padding=config.TOKENIZED_PADDING, embedding_dimensions=config.EMBEDDING_DIM, plot_model=True)
        self.model.build()