import keras as krs
import keras_contrib as krsa
import numpy as np
import os
from Callback.WarmupLRScheduler import WarmupLRScheduler
from Framework.Training import Training
from Model.NER.W2V_BiRNN_CRF import Model
from Loss.CRFLoss import CRFLoss


def main():
    # dataset
    train_x = np.ones([10, 20])
    train_y = np.ones([10, 20, 1])

    # model
    model_parameters = {
        "embedding_parameters": {
            "input_dim": 100,
            "output_dim": 16
        },
        "recurrent_parameters": {
            "units": 32
        },
        "crf_parameters": {
            "units": 3
        }
    }
    model_agent = Model(**model_parameters)
    model = model_agent.build_model()
    # print(model.predict(train_x))

    # loss
    loss_parameters = {}
    loss_agent = CRFLoss(**loss_parameters)
    loss_fn = loss_agent.build_loss()

    # optimizer
    optimizer = krs.optimizers.SGD(lr=0.1)

    # metrics
    metrics = [
        krsa.metrics.crf_accuracy
    ]

    # callbacks
    log_path = "Log"
    checkpoint_path = os.path.join("Checkpoints", "ner_model.cp")

    warmup_parameters = {
        "units": 32
    }
    scheduler = WarmupLRScheduler(**warmup_parameters)

    callbacks = [
        # log
        krs.callbacks.TensorBoard(log_path),
        krs.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=False),

        # lr
        krs.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        # scheduler,

        # training step
        krs.callbacks.EarlyStopping(patience=10, min_delta=1e-3)
    ]

    # trainer
    fit_parameters = {
        "epochs": 10,
        "batch_size": 2,
        "validation_split": 0.1
    }
    trainer = Training(
        train_x=train_x,
        train_y=train_y,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        callbacks=callbacks,
        fit_parameters=fit_parameters
    )
    trainer.run()


if __name__ == "__main__":
    main()
