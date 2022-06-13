class ForwardHook:
    def __init__(self, module):
        self.input = None
        self.output = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if isinstance(input, tuple):
            assert (
                len(input) == 1
            ), f"Expected length of the inputs tuple to be 1 in forward hook, but got {len(input)}"
            input = input[0]

        if isinstance(output, tuple):
            assert (
                len(output) == 1
            ), f"Expected length of the outputs tuple to be 1 in forward hook, but got {len(output)}"
            input = output[0]

        self.input = input.cpu().detach()
        self.output = output.cpu().detach()

    def close(self):
        self.hook.remove()
