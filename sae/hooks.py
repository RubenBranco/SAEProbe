class RecordingHookPoint:
    def __init__(self, model, layer_name):
        self.activation_store = []
        self.layer_name = layer_name
        layer = dict([*model.named_modules()])[layer_name]
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activation_store.append(output.detach().cpu())

    def reset_activation_store(self):
        self.activation_store = []

    def close(self):
        self.hook.remove()