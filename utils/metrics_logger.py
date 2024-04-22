import wandb


class MetricsLogger:
    def __init__(self, wandb_run: wandb.wandb_sdk.wandb_run.Run) -> None:
        self.wandb_run = wandb_run
        self.current_epoch = 0

        self.data = {}

    def log(self, key: str, value):
        """
        Log a datapoint.
        """

        # For Tensors
        if hasattr(value, "item"):
            value = value.item()

        # Handle the case where a new key suddenly starts being logged
        if key not in self.data:
            self.data[key] = {}

            for epoch in range(self.current_epoch - 1):
                self.data[key][epoch] = None

        if self.current_epoch not in self.data[key]:
            self.data[key][self.current_epoch] = []

        self.data[key][self.current_epoch].append(value)

    def log_dict(self, dict: dict):
        """
        Log many datapoints.
        """

        for k, v in dict.items():
            self.log(k, v)

    @property
    def current_epoch_data(self):
        data = {}

        for k, v in self.data.items():
            data[k] = v[self.current_epoch]

        return data

    def average_last(self):
        """
        When there are more datapoints per epoch (like the data logged in a the forward
        function of a module that is split it batches), we need to replace the list of values
        with their average, before logging them to wandb.
        """

        for k, v in self.data.items():
            if isinstance(v[self.current_epoch], list):
                values = [value for value in v[self.current_epoch] if value]

                if len(values):
                    self.data[k][self.current_epoch] = sum(values) / len(values)
                else:
                    self.data[k][self.current_epoch] = None

    def end_epoch(self):
        """
        Mark the end of an epoch.
        """

        self.average_last()

        self.wandb_run.log(self.current_epoch_data)
        self.current_epoch += 1
