import keras_contrib as krsa


class CRFLoss:
    def __init__(self, **kwargs):
        self.loss_fn = krsa.losses.crf_loss
        self.kwargs = kwargs

    def build_loss(self):
        return self.loss_fn
