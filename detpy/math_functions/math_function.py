class MathFunction:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __str__(self):
        return f"MathFunction: {self.name}"