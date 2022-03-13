from transformers import DistilBertTokenizerFast
import numpy as np
import os

from .config import Config as config
from .core import data_processing, model

class Aurras():

    def __init__(self):
        print('Initializing Aurras')

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_VARIANT)

    def train(self):
        """ Train a model on a loaded dataset """
        print('Training Aurras')

        self.model = model.Model(padding=config.TOKENIZED_PADDING, embedding_dimensions=config.EMBEDDING_DIM, plot_model=True)
        self.model.build()

        self.model.fit(self.dataset, config.EPOCHS, config.BATCH_SIZE, 1)

    def save(self):
        """ Save a model's weights to file """
        print('Saving Aurrass')

        self.model.save(config.MODEL_PATH)

    def load(self):
        """ Load a model from file """
        print('Loading pre-trained weights')

        self.model = model.Model(padding=config.TOKENIZED_PADDING, embedding_dimensions=config.EMBEDDING_DIM, plot_model=True)

        self.model.load(config.MODEL_PATH)

    def get_intent(self, prompt, prompt2):
        """ Determin the intent for a given prompt """
        print('Predicting an intent')

        tokenized1 = self.tokenizer(
			prompt,
			max_length=config.TOKENIZED_PADDING,
			padding='max_length',
			truncation=True,
			return_attention_mask=True,
			return_token_type_ids=False,
			return_tensors='np'
		)

        tokenized2 = self.tokenizer(
			prompt2,
			max_length=config.TOKENIZED_PADDING,
			padding='max_length',
			truncation=True,
			return_attention_mask=True,
			return_token_type_ids=False,
			return_tensors='np'
		)

        similarity = self.model.get_similarity(tokenized1['input_ids'], tokenized1['attention_mask'], tokenized2['input_ids'], tokenized2['attention_mask'])

        print(f" - p1: {prompt}")
        print(f" - p2: {prompt2}")
        print(f" - similarity: {similarity}")

    def generate_data_from_file(self):
        """ Generate a dataset from the raw provided data """
        print('Generating a dataset')

        json_data = data_processing.preprocess_intent_dataset(config.INTENT_SAMPLES, config.DATASET_PATH)
        self.dataset = data_processing.process_triplets(json_data)

        self.dataset = data_processing.tokenize_intent_dataset(self.dataset, self.tokenizer, config.TOKENIZED_PADDING)

        np.save(f'{config.DATASET_PATH}/dataset.npy', self.dataset)

    def load_data(self):
        """ Load a dataset from file """
        print('Loading a dataset')

        if os.path.isfile(f'{config.DATASET_PATH}/dataset.npy'):
            self.dataset = np.load(f'{config.DATASET_PATH}/dataset.npy')
        else:
            print("Couldn't load data from the file - generating a dataset")
            self.generate_data_from_file()

        print(self.dataset.shape)