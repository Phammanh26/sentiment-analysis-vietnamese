import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional


class SentimentModel:

    def __init__(self, model_path: Optional[str]) -> None:
        
        self.model_path = model_path
        self.model = None
        
    def loading(self) -> None:
        self.model = keras.models.load_model(self.model_path)
        return None

    def predict(self, text: str):
        return self.model.predict(text)
        