# from pathlib import Path

# from loguru import logger
# import typer

# from hsi_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR

# app = typer.Typer()


# import math

# import torch
# import torch.nn as nn
# import torch.nn.init as init


# class BaseModel(nn.Module):
#     def __init__(self, input_channels, n_classes):
#         super(BaseModel, self).__init__()
#         self.input_channels = input_channels
#         self.n_classes = n_classes

#     @staticmethod
#     def weight_init(m):
#         if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
#             init.uniform_(m.weight, -0.05, 0.05)
#             init.zeros_(m.bias)

# class HuEtAl(BaseModel):
#     def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
#         super(HuEtAl, self).__init__(input_channels, n_classes)
#         if kernel_size is None:
#             kernel_size = math.ceil(input_channels / 9)
#         if pool_size is None:
#             pool_size = math.ceil(kernel_size / 5)
#         self.conv = nn.Conv1d(1, 20, kernel_size)
#         self.pool = nn.MaxPool1d(pool_size)
#         self.features_size = self._get_final_flattened_size()
#         self.fc1 = nn.Linear(self.features_size, 100)
#         self.fc2 = nn.Linear(100, n_classes)
#         self.apply(self.weight_init)

#     def _get_final_flattened_size(self):
#         with torch.no_grad():
#             x = torch.zeros(1, 1, self.input_channels)
#             x = self.pool(self.conv(x))
#         return x.numel()

#     def forward(self, x):
#         x = x.squeeze(dim=-1).squeeze(dim=-1)
#         x = x.unsqueeze(1)
#         x = self.conv(x)
#         x = torch.tanh(self.pool(x))
#         x = x.view(-1, self.features_size)
#         x = torch.tanh(self.fc1(x))
#         x = self.fc2(x)
#         return x

# from collections import defaultdict


# class ModelRegistry:
#     _registry = defaultdict(dict)

#     @classmethod
#     def register(cls, model_name, model_cls):
#         cls._registry[model_name] = model_cls

#     @classmethod
#     def create_model(cls, model_name, input_channels, n_classes, **kwargs):
#         if model_name in cls._registry:
#             return cls._registry[model_name](input_channels, n_classes, **kwargs)
#         else:
#             raise ValueError(f"Unsupported model: {model_name}")


# @app.command()
# def main(
#     features_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
# ):
#     logger.info("Training some model...")

#     logger.success("Modeling training complete.")


# if __name__ == "__main__":
#     app()
