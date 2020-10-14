class Training:
    def __init__(self,
                 train_x,
                 train_y,
                 model,
                 loss_fn,
                 optimizer,
                 val_x=None,
                 val_y=None,
                 metrics=None,
                 callbacks=None,
                 compile_parameters=None,
                 fit_parameters=None,
                 **kwargs):

        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y

        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.callbacks = callbacks

        self.compile_parameters = self.get_default_compile_parameters()
        if compile_parameters is not None:
            self.compile_parameters.update(compile_parameters)

        self.fit_parameters = self.get_default_fit_parameters()
        if fit_parameters is not None:
            self.fit_parameters.update(fit_parameters)

        self.kwargs = kwargs

    def get_default_compile_parameters(self):
        parameters = {}

        return parameters

    def get_default_fit_parameters(self):
        parameters = {
            "epochs": 10,
            "batch_size": 512,
            "validation_split": 0.2
        }

        return parameters

    def verify_preparation_before_running(self):
        for name, attribute in \
            zip(["train_x", "train_y", "model", "loss_fn", "optimizer"],
                [self.train_x, self.train_y, self.model, self.loss_fn, self.optimizer]):
            if attribute is None:
                raise AttributeError("%s can't be None." % name)

    def run(self):
        self.verify_preparation_before_running()

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics,
            **self.compile_parameters
        )

        if self.val_x is None or self.val_y is None:
            self.model.fit(
                x=self.train_x,
                y=self.train_y,
                callbacks=self.callbacks,
                **self.fit_parameters
            )
        else:
            self.model.fit(
                x=self.train_x,
                y=self.train_y,
                validation_data=(self.val_x, self.val_y),
                callbacks=self.callbacks,
                **self.fit_parameters
            )
