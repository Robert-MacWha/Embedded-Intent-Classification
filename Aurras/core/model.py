import tensorflow as tf
from transformers import TFDistilBertModel

from .distance_layer import DistanceLayer
from .siamese_model import SiameseModel

class Model:

    def __init__(self, padding: int, embedding_dimensions: int, plot_model: bool):
        self.padding = padding
        self.embedding_dim = embedding_dimensions
        self.plot = plot_model

    def build(self):
        """ Build a new model & prepare it for training / predictions """
        
        self.embedding_model = self.__build_embedding_model()
        siamese_network = self.__build_siamese_network(self.embedding_model)
        self.siamese_model = SiameseModel(siamese_network)

        self.siamese_model.compile(optimizer=tf.keras.optimizers.Adam())

    def save(self, path: str):
        self.embedding_model.save_weights(f"{path}/embedding_model")

    def load(self, path: str):
        """ Load in a pre-trained model & prepare it for training / predictions """
        pass

    def fit(self, data: list, epochs: int, batch_size: int, verbose: int):
        """ Train the model """
        self.siamese_model.fit(data, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=verbose)

    def get_embedding(self, input_ids, input_attention):
        """ Embed a tokenized string """

        return self.embedding_model(input_ids, input_attention).numpy()

    def get_similarity(self, i1: tuple, i2: tuple):
        """ Return the cosine similarity of two tokenized strings """

        i1_embedding, i2_embedding = (self.embedding_model(i1[0], i1[1]), self.embedding_model(i2[0], i2[1]))
        return tf.keras.metrics.CosineSimilarity(i1_embedding, i2_embedding).numpy()

    def __build_embedding_model(self):

        weight_initializer = tf.keras.initializers.GlorotNormal(seed=42)

        #* inputs
        input_ids_layer       = tf.keras.layers.Input(shape=(self.padding), name='input_ids',       dtype='int32')
        input_attention_layer = tf.keras.layers.Input(shape=(self.padding), name='input_attention', dtype='int32')

        #* transformer
        transformer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        last_hidden_state = transformer([input_ids_layer, input_attention_layer], )[0] # (batch_size, sequence_length, hidden_size=768)

        #* outputs
        cls_token = last_hidden_state[:, 0, :]

        x = tf.keras.layers.BatchNormalization()(cls_token)

        x = tf.keras.layers.Dense(
            512, 
            activation='relu',
            kernel_initializer=weight_initializer,
            kernel_constraint=None,
            bias_initializer='zeros' 
        )(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        embedded_output = tf.keras.layers.Dense(
            self.embedding_dim,
            activation='softmax',
            kernel_initializer=weight_initializer,
            kernel_constraint=None,
            bias_initializer='zeros',
            name='embedding'
        )(x)

        embedding = tf.keras.models.Model([input_ids_layer, input_attention_layer], embedded_output)    

        for layer in embedding.layers[:3]:
            layer.trainable = False

        if self.plot:
            tf.keras.utils.plot_model(embedding, to_file='embedding.png', show_shapes=True, show_layer_names=True)

        return embedding

    def __build_siamese_network(self, embedding_model):
        
        #* input
        input = tf.keras.layers.Input(shape=(6, self.padding), name='input', dtype='int32')

        anchor_ids_layer       = input[:,0,:]
        anchor_attention_layer = input[:,1,:]

        positive_ids_layer       = input[:,2,:]
        positive_attention_layer = input[:,3,:]

        negative_ids_layer       = input[:,4,:]
        negative_attention_layer = input[:,5,:]

        #* output
        distances = DistanceLayer()(
            embedding_model([anchor_ids_layer, anchor_attention_layer]),     # anchor
            embedding_model([positive_ids_layer, positive_attention_layer]), # positive
            embedding_model([negative_ids_layer, negative_attention_layer])  # negative
        )

        siamese_network = tf.keras.models.Model(
            input,
            distances
        )

        if self.plot:
            tf.keras.utils.plot_model(siamese_network, to_file='siamese.png', show_shapes=True, expand_nested=True, show_layer_names=True)

        return siamese_network
