from transformers import DistilBertTokenizerFast
import tensorflow as tf
import numpy as np
import time
import json
import os

from .config import Config as config
from .core import data_processing, model

class Aurras():

    def __init__(self):
        print('Initializing Aurras')

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_VARIANT)
        self.intent_dict = {}
        self.model = None

    """ Interactions """
    def add_intent(self, samples: list, name: str = None):
        """
            Adds a new intent to Aurras's memory
            
            Inputs:
             - samples: list of strings coresponding to the intent
             - name (optional): name for this intent
        """

        if name == None:
            name = f"__{round(time.time())}__"

            if name in self.intent_dict:
                print("ERROR: failed to add intent.  Intent already exists")

        if self.model == None:
            print("ERROR: Failed to add intent.  An embedding model must first be loaded")
            return -1

        # calculate the embedding for each intent & take the average
        embeddings = []
        for s in samples:
            (ids, attention) = self.__tokenize(s)
            e = self.model.get_embedding(ids, attention)
            embeddings.append(e)

        avg = np.mean(np.array(embeddings), axis=0)
        
        # add the intent
        intent = {
            'embedding': avg.tolist(),
            'samples': samples,
        }

        self.intent_dict[name] = intent

    def add_sample_to_intent(self, sample: str, intent_name: str):
        """
            Add a sample to an intent

            Inputs:
             - sample: string sample
             - intent_name: name of the intent
        """

        if not intent_name in self.intent_dict:
            print('ERROR: An intent with that name does not net exist')
            return -1

        # calculate sample embedding & update average based on weight of new sample
        (ids, attention) = self.__tokenize(sample)
        sample_e = self.model.get_embedding(ids, attention)
        
        intent = self.intent_dict[intent_name]
        sample_count = len(intent['samples'])

        e_weight = (1 / (sample_count + 1))
        s_weight = (1 - e_weight)

        # multiply each element by that list's weight
        e_weighted = [e * e_weight for e in sample_e]
        s_weighted = [s * s_weight for s in intent['embedding']]

        # sum of the two lists & save that
        embedding = [sum(v) for v in zip(e_weighted, s_weighted)]
        self.intent_dict[intent_name]['embedding'] = embedding
        self.intent_dict[intent_name]['samples'].append(sample)

    """ Training process """
    def save(self):
        """ Save aurras """
        print('Saving Aurras')
        
        self.model.save(config.MODEL_PATH)
        json.dump(self.intent_dict, open(config.INTENTS_PATH, 'w+'))

    def load(self):
        """ Load aurras from file """
        print('Loading pre-trained weights')

        try:
            self.model = model.Model(padding=config.TOKENIZED_PADDING, embedding_dimensions=config.EMBEDDING_DIM, plot_model=True)
            self.model.load(config.MODEL_PATH)
        except:
            print("ERROR: Failed to load pre-trained model")

        try:
            self.intent_dict = json.load(open(config.INTENTS_PATH))
        except:
            print("ERROR: Failed to load intents dict")

    def train(self):
        """ Train a model on a loaded dataset """
        print('Training Aurras')

        self.model = model.Model(padding=config.TOKENIZED_PADDING, embedding_dimensions=config.EMBEDDING_DIM, plot_model=True)
        self.model.build()

        self.model.fit(self.dataset, config.EPOCHS, config.BATCH_SIZE, 1)

    """ Data management """
    def get_intent(self, prompt):
        """ Determin the intent for a given prompt """
        print('Predicting an intent')

        (ids, attention) = self.__tokenize(prompt)
        embedding = self.model.get_embedding(ids, attention)

        max_similarity = config.MIN_INTENT_SIMILARITY
        intent = None
        for key, value in self.intent_dict.items():

            cosine_similarity = tf.keras.metrics.CosineSimilarity()
            similarity = cosine_similarity(embedding, np.array(value['embedding'])).numpy()

            print(similarity)
            if similarity > max_similarity:
                max_similarity = similarity
                intent = key
        
        if intent == None:
            print("WARNING: No valid intent was found for this prompt")
            return -1

        return {"intent": intent, "Confidence": similarity}

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

    def __tokenize(self, s: str):
        tokenized =  self.tokenizer(
			s,
			max_length=config.TOKENIZED_PADDING,
			padding='max_length',
			truncation=True,
			return_attention_mask=True,
			return_token_type_ids=False,
			return_tensors='np'
		)

        return (tokenized['input_ids'], tokenized['attention_mask'])