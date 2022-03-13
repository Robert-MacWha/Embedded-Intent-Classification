import tensorflow as tf
from transformers import TFDistilBertModel

class Model:

    def __init__(self, padding: int, plot_model: bool):
        self.padding = padding
        self.plot = plot_model

    def build(self):
        """ Build a new model & prepare it for training / predictions """

        input = tf.keras.layers.Input(shape=(4, self.padding), name='input', dtype='int32')
        
         #* inputs
        input_ids_layer_1       = input[:,0,:]
        input_attention_layer_1 = input[:,1,:]

        input_ids_layer_2       = input[:,2,:]
        input_attention_layer_2 = input[:,3,:]

        #* transformer
        transformer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        last_hidden_state_1 = transformer([input_ids_layer_1, input_attention_layer_1], )[0] # (batch_size, sequence_length, hidden_size=768)
        last_hidden_state_2 = transformer([input_ids_layer_2, input_attention_layer_2], )[0] # (batch_size, sequence_length, hidden_size=768)

        #* outputs
        cls_token_1 = last_hidden_state_1[:, 0, :]
        cls_token_2 = last_hidden_state_2[:, 0, :]

        x = tf.keras.layers.concatenate([cls_token_1, cls_token_2])

        x = tf.keras.layers.Dense(
            512, 
            activation='relu',
            kernel_constraint=None,
        )(x)

        x = tf.keras.layers.Dropout(0.2)(x)

        similarity = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_constraint=None,
            name='classifier'
        )(x)

        self.model = tf.keras.models.Model(input, similarity)
        self.model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.BinaryAccuracy()])

        for layer in self.model.layers[:6]:
            layer.trainable = False

        if self.plot:
            tf.keras.utils.plot_model(self.model, to_file='intent_similarity_model.png', show_shapes=True, show_layer_names=True)

    def fit(self, Xs: list, ys: list, epochs: int, batch_size: int, verbose: int):
        """ Train the model """
        self.model.fit(Xs, ys, epochs=epochs, validation_split=0.05, batch_size=batch_size, verbose=verbose)

    def save(self, path: str):
        self.model.save(f"{path}/model.h5")