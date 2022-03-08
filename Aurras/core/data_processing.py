import glob
import random
from pathlib import Path

def preprocess_intent_dataset(samples_per_intent: int=5, raw_path: str='dataset/raw'):
    """
        Generate a dataset for the triplet intent classification model

        Inputs:
         - samples_per_intent: number of samples to generate for each intent
         - raw_path: path to the parent folder of the raw dataset (must contain intent and entity subdirectories)
    """

    """
    #* load the intents & entities into memory
    intent_files = glob.glob(f'{raw_path}/intents/*.intent', recursive=True)
    entity_files = glob.glob(f'{raw_path}/entities/*.entity', recursive=True)

    raw_intents = {}
    for file in intent_files:
        intent_name = Path(file).stem
        samples = [i.lower() for i in open(file).read().splitlines() if not i.startswith('#')]
        raw_intents[intent_name] = samples

    entities = {}
    for file in entity_files:
        name = Path(file).stem
        samples = [e.lower() for e in open(file).read().splitlines() if not e.startswith('#')]
        entities[name] = samples

    #* generate all permutations
    permuted_intents = {}
    for key in raw_intents:
        intent_templates = raw_intents[key]

        # generate x permutations for each intent category
        permuted_intents[key] = []
        while len(permuted_intents[key]) < samples_per_intent:
            intent = random.choice(intent_templates)

            # fill the intent's entity slots
            intent = intent.split()
            for i, word in enumerate(intent):
                if word.startswith('{') and word.endswith('}'):
                    

            permuted_intents[key].append(intent)
        break

    print(permuted_intents)
    """