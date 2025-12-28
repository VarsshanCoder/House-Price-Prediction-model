class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register(self, name, model):
        self.models[name] = model

    def get_model(self, name):
        if name not in self.models:
            raise ValueError(f"Model {name} not found in registry.")
        return self.models[name]

    def list_models(self):
        return list(self.models.keys())

registry = ModelRegistry()
