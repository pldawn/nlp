import keras as krs
from Layer.CRF import CRF
from Layer.Embedding import W2VEmbedding
from Layer.Recurrent import Recurrent


class Model:
    def __init__(self,
                 embedding_parameters=None,
                 recurrent_parameters=None,
                 crf_parameters=None,
                 **kwargs):
        self.embedding_parameters = self.get_default_embedding_parameters()
        if embedding_parameters is not None:
            self.embedding_parameters.update(embedding_parameters)

        self.recurrent_parameters = self.get_default_recurrent_parameters()
        if recurrent_parameters is not None:
            self.recurrent_parameters.update(recurrent_parameters)

        self.crf_parameters = self.get_default_crf_parameters()
        if crf_parameters is not None:
            self.crf_parameters.update(crf_parameters)

        self.kwargs = kwargs

    def build_model(self):
        self.verify_parameters_compatibility()
        model = krs.Sequential()

        # build embedding layers and add it to model
        embedding_layers = W2VEmbedding(**self.embedding_parameters).build_layers()
        for layer in embedding_layers:
            model.add(layer)

        # build recurrent layers and add them to model
        recurrent_layers = Recurrent(**self.recurrent_parameters).build_layers()
        for layer in recurrent_layers:
            model.add(layer)

        # build crf layers and add it to model
        crf_layers = CRF(**self.crf_parameters).build_layers()
        for layer in crf_layers:
            model.add(layer)

        return model

    def verify_parameters_compatibility(self):
        if self.recurrent_parameters["layer_nums"] > 1:
            if not self.recurrent_parameters["return_sequences"]:
                raise ValueError("when layers_nums greater than 1, return sequences should be True.")

    def get_default_embedding_parameters(self):
        parameters = {
            "input_dim": 10000,
            "output_dim": 256,
            "mask_zero": True,
            "w2v_embeddings_path": None
        }

        return parameters

    def get_default_recurrent_parameters(self):
        parameters = {
            "units": 256,
            "layer_nums": 2,
            "rnn_type": "lstm",
            "return_sequences": True
        }

        return parameters

    def get_default_crf_parameters(self):
        parameters = {
            "units": 2,
            "sparse_target": True,
            "input_dim": None
        }

        return parameters
