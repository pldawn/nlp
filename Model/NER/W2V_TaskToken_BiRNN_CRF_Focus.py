import tensorflow as tf
import keras as krs
from Layer.CRF import CRF
from Model.NER import W2V_BiRNN_CRF
from Layer.Embedding import Embedding, W2VEmbedding
from Layer.Recurrent import Recurrent


class Model(W2V_BiRNN_CRF.Model):
    def __init__(self,
                 word_embedding_parameters=None,
                 task_embedding_parameters=None,
                 recurrent_parameters=None,
                 crf_parameters=None,
                 **kwargs):
        self.task_embedding_parameters = self.get_default_task_embedding_parameters()
        if task_embedding_parameters is not None:
            self.task_embedding_parameters.update(task_embedding_parameters)

        super(Model, self).__init__(
            word_embedding_parameters,
            recurrent_parameters,
            crf_parameters,
            **kwargs
        )

    def build_model(self):
        self.verify_parameters_compatibility()

        word_tokens = krs.layers.Input()
        task_token = krs.layers.Input()

        # build embedding layers and add it to model
        word_embedding_layers = W2VEmbedding(**self.embedding_parameters).build_layers()
        task_embedding_layers = Embedding(**self.task_embedding_parameters).build_layers()

        word_hidden = word_tokens
        task_hidden = task_token

        for layer_num in range(len(word_embedding_layers)):
            word_hidden = word_embedding_layers[layer_num](word_hidden)
            task_hidden = task_embedding_layers[layer_num](task_hidden)

        hidden_state = tf.concat([task_hidden, word_hidden], axis=1)

        # build recurrent layers and add them to model
        recurrent_layers = Recurrent(**self.recurrent_parameters).build_layers()
        for layer in recurrent_layers:
            hidden_state = layer(hidden_state)

        # build crf layers and add it to model
        crf_layers = CRF(**self.crf_parameters).build_layers()
        for layer in crf_layers:
            hidden_state = layer(hidden_state)

        model = krs.models.Model(inputs=[word_tokens, task_token], outputs=[hidden_state])

        return model

    def get_default_task_embedding_parameters(self):
        parameters = {
            "input_dim": 10000,
            "output_dim": 256,
            "mask_zero": False,
            "embeddings_initializer": "uniform"
        }

        return parameters
