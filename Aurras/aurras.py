from transformers import DistilBertTokenizerFast
import numpy as np
import time
import json
import os

from .config import Config as CONFIG
from .data import data_processing
from .model import model

class Aurras():

    def __init__(self):
        print('Initializing Aurras')

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(CONFIG.MODEL_VARIANT)
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
        self.intent_dict[name] = []
        for s in samples:
            (ids, attention) = self.__tokenize(s)
            e = self.model.get_embedding(ids, attention)
            
            self.intent_dict[name].append({
                'prompt': s,
                'embedding': e.tolist()
            })

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
        e = self.model.get_embedding(ids, attention)
        
        self.intent_dict[intent_name].append({
            'prompt': sample,
            'embedding': e
        })

    def get_intent(self, prompt):
        """ Determin the intent for a given prompt using k mean clustering """
        print('Predicting an intent')

        (ids, attention) = self.__tokenize(prompt)
        prompt_embedding = self.model.get_embedding(ids, attention)

        #? k mean clustering to determin the intent
        distances = [] # list of lists [intent, distance to prompt]

        for key, value in self.intent_dict.items():
            for x in value:
                distance = np.sum(np.square(prompt_embedding - np.array(x['embedding'])), -1)

                # only add the point if it's closer than some threshold
                if distance < CONFIG.MAX_INTENT_DISTANCE:
                    distances.append([key, distance])

        # fail if no prompts are similar enough
        if len(distances) == 0:
            print("ERROR: No similar prompts found, unknown intent")
            return -1

        # sort and cull
        distances.sort(key=lambda x:x[1])
        distances = distances[:CONFIG.INTENT_K]

        # count up how many times each intent shows up
        intent_frequency = {}
        for s in distances:
            if s[0] in intent_frequency:
                intent_frequency[s[0]].append(s)
            else:
                intent_frequency[s[0]] = [s]

        intent = ""
        max_frequency = 0
        min_distance = 0
        for key, value in intent_frequency.items():
            if len(value) > max_frequency:
                intent = key
                max_frequency = len(value)
                min_distance = min([v[1] for v in value])

        return { "intent": intent, "min_distance": min_distance }

    """ Training process """
    def save(self):
        """ Save aurras """
        print('Saving Aurras')
        
        self.model.save(CONFIG.MODEL_PATH)
        json.dump(self.intent_dict, open(CONFIG.INTENTS_PATH, 'w+'))

    def load(self):
        """ Load aurras from file """
        print('Loading pre-trained weights')

        try:
            self.model = model.Model(padding=CONFIG.TOKENIZED_PADDING, embedding_dimensions=CONFIG.EMBEDDING_DIM, plot_model=False)
            self.model.load(CONFIG.MODEL_PATH)
        except:
            print("ERROR: Failed to load pre-trained model")

        try:
            self.intent_dict = json.load(open(CONFIG.INTENTS_PATH))
        except:
            print("ERROR: Failed to load intents dict")

    def train(self):
        """ Train a model on a loaded dataset """
        print('Training Aurras')

        self.model = model.Model(padding=CONFIG.TOKENIZED_PADDING, embedding_dimensions=CONFIG.EMBEDDING_DIM, plot_model=True)
        self.model.build()

        self.model.fit(self.dataset, CONFIG.EPOCHS, CONFIG.BATCH_SIZE, 1)

    """ Data management """
    def generate_data_from_clinc(self):
        """ Generate a dataset from the clinc 150 dataset """
        print("Generating clinc 150 dataset")

        # load in json data from the clinc 150 dataset and convert it into Aurras' json format 
        # ? dict with intents as keys & lists of prompts as values
        json_raw  = json.load(open(f'{CONFIG.DATASET_PATH}/clinc150/data.json'))
        raw_list  = json_raw['train'] + json_raw['test'] + json_raw['val']
        json_data = {}
        
        for x in raw_list:
            prompt = x[0]
            intent = x[1]

            if intent == 'oos': # avoid adding oos since we want to classify any out of scope intents
                continue

            if intent in json_data:
                json_data[intent].append(prompt)
            else:
                json_data[intent] = [prompt]

        self.dataset = data_processing.process_triplets(json_data)
        self.dataset = data_processing.tokenize_triplet_intent_dataset(self.dataset, self.tokenizer, CONFIG.TOKENIZED_PADDING)
        np.save(f'{CONFIG.DATASET_PATH}/dataset.npy', self.dataset)

    def generate_data_from_raw(self):
        """ Generate a dataset from the raw provided data """
        print('Generating raw dataset')

        json_data    = data_processing.preprocess_intent_dataset(CONFIG.INTENT_SAMPLES, CONFIG.DATASET_PATH)
        self.dataset = data_processing.process_triplets(json_data)
        self.dataset = data_processing.tokenize_triplet_intent_dataset(self.dataset, self.tokenizer, CONFIG.TOKENIZED_PADDING)
        np.save(f'{CONFIG.DATASET_PATH}/dataset.npy', self.dataset)

    def load_data(self):
        """ Load a dataset from file """
        print('Loading a dataset')

        if os.path.isfile(f'{CONFIG.DATASET_PATH}/dataset.npy'):
            self.dataset = np.load(f'{CONFIG.DATASET_PATH}/dataset.npy')
        else:
            print("Couldn't load data from the file - generating a dataset")
            self.generate_data_from_file()

        print(self.dataset.shape)

    """ Private functions """
    def __tokenize(self, s: str):
        tokenized =  self.tokenizer(
			s,
			max_length=CONFIG.TOKENIZED_PADDING,
			padding='max_length',
			truncation=True,
			return_attention_mask=True,
			return_token_type_ids=False,
			return_tensors='np'
		)

        return (tokenized['input_ids'], tokenized['attention_mask'])