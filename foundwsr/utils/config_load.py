import yaml
import importlib
import re

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(e)

class Config:
    def __init__(self, **kwargs):
        self.scientific_notation_pattern = re.compile(r"[+-]?\d+(\.\d+)?[eE][+-]?\d+")
        for key, value in kwargs.items():
            if isinstance(value, list) or isinstance(value, dict):
                setattr(self, key, self.recursion(value))
            elif isinstance(value, str):
                if value.startswith("nn."):
                    module = importlib.import_module("torch.nn")
                    class_name = value.split("nn.")[-1]
                    act_layer = getattr(module, class_name)
                    setattr(self, key, act_layer)
                elif self.scientific_notation_pattern.match(value):
                    setattr(self, key, float(value))
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
    
    def recursion(self, value):
        if isinstance(value, list):
            for i, ele in enumerate(value):
                value[i] = self.recursion(ele)
            return value
        elif isinstance(value, dict):
            for key, val in value.items():
                value[key] = self.recursion(val)
            return value
        elif isinstance(value, str):
            if self.scientific_notation_pattern.match(value):
                return float(value)
            else:
                return value
        else:
            return value