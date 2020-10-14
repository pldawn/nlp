import tensorflow as tf
import keras as krs


class FocusLossForSingleTask:
    def __init__(self, alpha=4.0, gama=0.5, sparse_target=True, **kwargs):
        self.alpha = alpha
        self.gama = gama
        self.sparse_target = sparse_target
        self.kwargs = kwargs

        if self.sparse_target:
            self.core_loss_fn = krs.losses.SparseCategoricalCrossentropy(**kwargs)
        else:
            self.core_loss_fn = krs.losses.CategoricalCrossentropy(**kwargs)

    def __call__(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.convert_to_tensor(y_pred)

        if self.kwargs.get("from_logits", False):
            y_pred_prob = krs.activations.softmax(y_pred)
            print(y_pred_prob)
        else:
            y_pred_prob = y_pred

        if self.sparse_target:
            if y_true.ndim > 1:
                y_true = tf.squeeze(y_true)

            sample_weight = [1 - y_pred_prob[i][y_true[i]] for i in range(len(y_true))]

        else:
            sample_weight = [1 - tf.reduce_sum(tf.cast(y_true[i], y_pred_prob.dtype) * y_pred_prob[i])
                             for i in range(len(y_true))]

        sample_weight = tf.convert_to_tensor(sample_weight)
        sample_weight = self.alpha * sample_weight ** self.gama

        loss = self.core_loss_fn(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

        return loss


if __name__ == "__main__":
    agent = FocusLossForSingleTask()
    print(agent([1, 0], [[0.9, 0.1], [0.7, 0.3]]))
