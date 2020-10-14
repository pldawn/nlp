import keras as krs


class Recurrent:
    def __init__(self,
                 units,
                 bidirectional=True,
                 layer_nums=1,
                 rnn_type="lstm",  # valid value: lstm, gru
                 **kwargs):
        self.units = units
        self.bidirectional = bidirectional
        self.layer_nums = layer_nums
        self.rnn_type = rnn_type
        self.kwargs = kwargs

    def build_layers(self):
        layers = []

        for _ in range(self.layer_nums):
            if self.rnn_type == "lstm":
                layer = krs.layers.LSTM(self.units, **self.kwargs)
            elif self.rnn_type == "gru":
                layer = krs.layers.GRU(self.units, **self.kwargs)
            else:
                raise KeyError("rnn_type has invalid value: %s" % self.rnn_type)

            if self.bidirectional:
                layer = krs.layers.Bidirectional(layer)

            layers.append(layer)

        return layers


if __name__ == "__main__":
    agent = Recurrent(10, rnn_type="gru")
    recurrent = agent.build_layers()
