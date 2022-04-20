class Config():
    INTENT_SAMPLES = 500
    TOKENIZED_PADDING = 128
    EMBEDDING_DIM = 64
    DATASET_PATH = 'dataset'
    MODEL_PATH = 'save/model'
    INTENTS_PATH = 'save/intents.json'
    MODEL_VARIANT = 'distilbert-base-uncased'

    EPOCHS = 2
    BATCH_SIZE = 32

    # how many neighbor embeddings to consider for k mean clustering
    INTENT_K = 5
    # how close a prompt needs to be to an intent to be considered classified
    MAX_INTENT_DISTANCE = 0.5