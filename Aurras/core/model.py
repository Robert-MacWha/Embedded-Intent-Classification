import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizerFast

from .distance_layer import DistanceLayer

class Model:

    def __init__(self, padding: int):
        
        self.padding = padding

    def build_model(self, embedding_dimensions: int=32):
        
        #* input
        anchor_ids_layer       = tf.keras.layers.Input(shape=(self.padding), name='anchor_ids',       dtype='int32')
        anchor_attention_layer = tf.keras.layers.Input(shape=(self.padding), name='anchor_attention', dtype='int32')

        positive_ids_layer       = tf.keras.layers.Input(shape=(self.padding), name='positive_ids',       dtype='int32')
        positive_attention_layer = tf.keras.layers.Input(shape=(self.padding), name='positive_attention', dtype='int32')

        negative_ids_layer       = tf.keras.layers.Input(shape=(self.padding), name='negative_ids',       dtype='int32')
        negative_attention_layer = tf.keras.layers.Input(shape=(self.padding), name='negative_attention', dtype='int32')

        #* hidden
        self.embedding_model = self.build_embedding_model(embedding_dimensions)

        #* output
        distances = DistanceLayer()(
            self.embedding_model([anchor_ids_layer, anchor_attention_layer]),
            self.embedding_model([positive_ids_layer, positive_attention_layer]),
            self.embedding_model([negative_ids_layer, negative_attention_layer])
        )

        self.siamese_model = tf.keras.models.Model(
            [anchor_ids_layer, anchor_attention_layer, positive_ids_layer, positive_attention_layer, negative_ids_layer, negative_attention_layer],
            distances
        )

        #* TEST: print out the model summary
        tf.keras.utils.plot_model(self.siamese_model, to_file='siamese.png', show_shapes=True, show_layer_names=True)

    def build_embedding_model(self, embedding_dimensions: int):

        weight_initializer = tf.keras.initializers.GlorotNormal(seed=42)

        # build the model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

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
            embedding_dimensions,
            activation='softmax',
            kernel_initializer=weight_initializer,
            kernel_constraint=None,
            bias_initializer='zeros',
            name='embedding'
        )(x)

        embedding = tf.keras.models.Model([input_ids_layer, input_attention_layer], embedded_output)

        for layer in embedding.layers[:3]:
            layer.trainable = False

        tf.keras.utils.plot_model(embedding, to_file='embedding.png', show_shapes=True, show_layer_names=True)

        return embedding
