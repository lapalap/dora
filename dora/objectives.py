class Objective:
    def __init__(self, layer_number=0, channel_number=0) -> None:
        """Objective function builder for channel optimizations.
        "Channel" is the same as "neuron" in this context.

        Args:
            layer_number (int, optional): the layer index whose outputs are to be used for the objective. Defaults to 0.
            channel_number (int, optional): the channel index/neuron index to be optimized. Defaults to 0.
        """
        self.layer_number = layer_number
        self.channel_number = 0

    def __call__(self, layer_outputs):
        return self.objective(layer_outputs=layer_outputs)

    def objective(self, layer_outputs):
        raise NotImplementedError


class ChannelObjective(Objective):
    def __init__(self, layer_number=0, channel_number=0) -> None:
        """Objective function builder for channel optimizations.
        "Channel" is the same as "neuron" in this context.

        Args:
            layer_number (int, optional): the layer index whose outputs are to be used for the objective. Defaults to 0.
            channel_number (int, optional): the channel index/neuron index to be optimized. Defaults to 0.
        """
        super().__init__(layer_number=layer_number, channel_number=channel_number)

    def objective(self, layer_outputs):
        loss = layer_outputs[self.layer_number][self.channel_number].mean()
        return -loss
