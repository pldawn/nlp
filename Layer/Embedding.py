import keras as krs
from gensim.models import Word2Vec
import os


class Embedding:
    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 **kwargs
                 ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.kwargs = kwargs

    def build_layers(self):
        layers = []
        layer = krs.layers.Embedding(self.input_dim, self.output_dim,
                                     embeddings_initializer=self.embeddings_initializer, **self.kwargs)
        layers.append(layer)

        return layers


class W2VEmbedding(Embedding):
    def __init__(self,
                 input_dim,
                 output_dim,
                 w2v_embeddings_path=None,  # str
                 **kwargs
                 ):
        if w2v_embeddings_path is None:
            self.w2v_embeddings = self.get_default_w2v_embeddings()
        else:
            self.w2v_embeddings = self.get_w2v_embeddings(w2v_embeddings_path)

        embeddings_initializer = krs.initializers.constant(self.w2v_embeddings)
        # noinspection PyTypeChecker
        super(W2VEmbedding, self).__init__(input_dim, output_dim, embeddings_initializer, **kwargs)

    def get_default_w2v_embeddings(self):
        default_path = os.path.join("Resources", "embeddings_w2v_default.model")
        embeddings = self.get_w2v_embeddings(default_path)

        return embeddings

    def get_w2v_embeddings(self, w2v_embeddings_path):
        model = Word2Vec.load(w2v_embeddings_path)
        embeddings = model.wv.vectors

        return embeddings


if __name__ == "__main__":
    agent = W2VEmbedding(10, 20)
    embedding = agent.build_layers()
