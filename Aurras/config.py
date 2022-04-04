class Config():
    INTENT_SAMPLES = 500
    TOKENIZED_PADDING = 128
    EMBEDDING_DIM = 32
    DATASET_PATH = 'dataset'
    MODEL_PATH = 'save/model'
    INTENTS_PATH = 'save/intents.json'
    MODEL_VARIANT = 'distilbert-base-uncased'

    EPOCHS = 5
    BATCH_SIZE = 16

    # how close a prompt needs to be to an intent to be considered classified
    MIN_INTENT_SIMILARITY = 0.2