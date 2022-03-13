class Config():
    INTENT_SAMPLES = 500
    TOKENIZED_PADDING = 128
    EMBEDDING_DIM = 32
    DATASET_PATH = 'dataset'
    MODEL_PATH = 'model'
    MODEL_VARIANT = 'distilbert-base-uncased'

    EPOCHS = 5
    BATCH_SIZE = 16