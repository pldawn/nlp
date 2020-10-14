import keras_contrib as krsa


class CRF:
    def __init__(self,
                 units,
                 sparse_target=True,
                 input_dim=None,
                 **kwargs):
        self.units = units
        self.sparse_target = sparse_target
        self.input_dim = input_dim
        self.kwargs = kwargs

    def build_layers(self):
        layers = []
        layer = krsa.layers.CRF(units=self.units, sparse_target=self.sparse_target,
                                input_dim=self.input_dim, **self.kwargs)
        layers.append(layer)

        return layers


if __name__ == "__main__":
    agent = CRF(10)
    crf = agent.build_layers()
