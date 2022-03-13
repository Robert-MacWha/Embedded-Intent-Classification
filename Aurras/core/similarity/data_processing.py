import glob
import json
import random
import numpy as np
from pathlib import Path
def preprocess_intent_dataset(samples_per_intent: int, data_path: str):
	"""
		Generate a dataset for the triplet intent classification model

		Inputs:
		 - samples_per_intent: number of samples to generate for each intent
		 - data_path: path to the parent folder of the raw dataset
	"""

	#* load the intents & entities into memory
	intent_files = glob.glob(f'{data_path}/raw/intents/*.intent', recursive=True)
	entity_files = glob.glob(f'{data_path}/raw/entities/*.entity', recursive=True)

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
	ATTEMPTS_BEFORE_FAIL = 50

	permuted_intents = {}
	for key in raw_intents:
		intent_templates = raw_intents[key]

		# generate x permutations for each intent category
		permuted_intents[key] = []

		fails = 0
		while len(permuted_intents[key]) < samples_per_intent and fails < ATTEMPTS_BEFORE_FAIL:
			intent = random.choice(intent_templates)

			# fill the intent's entity slots
			intent = intent.split()
			for i, word in enumerate(intent):
				if word.startswith('{') and word.endswith('}'):
					entity_label = word[1:-1]
					if entity_label in entities:
						intent[i] = random.choice(entities[entity_label])
					else:
						print(f'ERROR: Failed to add an intent template, unrecognised entity. Invalid intent template: {"".join(intent)}')
						continue
			
			# re-convert the intent into a single string, clean them up a little, & add it to the dataset
			intent = " ".join(intent)
			intent = intent.lstrip().rstrip()

			if not intent in permuted_intents[key]:
				permuted_intents[key].append(intent)
				fails = 0
			else:
				fails += 1

	# debugging info on how many permutations were generated
	print(f"The following intents were not able to generate {samples_per_intent} samples:")
	for key in permuted_intents:
		if len(permuted_intents[key]) < samples_per_intent:
			print(f" - {key}: {len(permuted_intents[key])}")

	return permuted_intents

def process_pairs(raw_sorted_data: json):
	""" Convert raw json data into batches of pairs """

	raw_data = []
	for key in raw_sorted_data:
		for v in raw_sorted_data[key]:
			raw_data.append(v)

	Xs = []
	ys = []

	for key in raw_sorted_data:

		key_samples = raw_sorted_data[key]

		# for each anchor prompt choose a random prompt from the same key as the anchor and a random prompt from the entire set as the different key
		for i in range(len(key_samples)):
			
			anchor = key_samples[i]
			
			positive_id = random.randint(0, len(key_samples) - 1)
			positive = key_samples[positive_id]

			negative_id = random.randint(0, len(raw_data) - 1)
			negative = raw_data[negative_id]

			Xs.append((anchor, positive))
			ys.append(1)

			Xs.append((anchor, negative))
			ys.append(0)

	return np.array(Xs), np.array(ys)

def tokenize_intent_dataset(triplet_data: list, tokenizer: object, padding: int):
	"""
		Tokenizes the pairs of text data & return them
	"""

	tokenized_data = []

	for sample in triplet_data:

		raw_tokenized_sample = tokenizer(
			[sample[0], sample[1]],
			max_length=padding,
			padding='max_length',
			truncation=True,
			return_attention_mask=True,
			return_token_type_ids=False,
			return_tensors='np'
		)

		tokenized_sample = (
			raw_tokenized_sample['input_ids'][0], raw_tokenized_sample['attention_mask'][0], 
			raw_tokenized_sample['input_ids'][1], raw_tokenized_sample['attention_mask'][1]
		)

		tokenized_data.append(tokenized_sample)

	return np.array(tokenized_data)