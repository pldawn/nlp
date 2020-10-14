import tensorflow as tf
import keras as krs


class WarmupLRScheduler(krs.callbacks.LearningRateScheduler):
    def __init__(self, units=512, warmup_epoch=5, verbose=0):
        super(WarmupLRScheduler, self).__init__(
            schedule=self.warmup_schedule,
            verbose=verbose
        )
        self.units = float(units)
        self.warmup_epoch = float(warmup_epoch)

    # learning_rate = (units * -0.5) * min(step_num ** -0.5, step_num * warm_up_steps ** -1.5))
    def warmup_schedule(self, epoch):
        epoch = float(epoch)

        args1 = tf.math.rsqrt(epoch)
        args2 = epoch * (self.warmup_epoch ** -1.5)
        args3 = tf.math.rsqrt(self.units)

        lr = args3 * tf.math.minimum(args1, args2)

        return lr


if __name__ == "__main__":
    agent = WarmupLRScheduler(100)
    agent.warmup_schedule(10)
