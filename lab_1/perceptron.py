import numpy as np
from typing import List
from loader import TrainData


class Neuron:
    def __init__(self, num_weights: int) -> None:
        self.weights = self.__get_weights(num_weights)

    @staticmethod
    def __get_weights(num_weights: int) -> np.array:
        return np.random.uniform(-1, 1, num_weights)

    def train(self, train_data: List[TrainData], learning_rate: float = 0.1, epochs: int = 100) -> None:
        for epoch in range(epochs):
            for item in train_data:
                signal_sum = self.__get_sum_for_weights(item.data)
                output = self.__get_activation_function(signal_sum)
                error = self.__get_error(item.target, output)
                self.__normalize_weights(error, item.data, learning_rate)

    def predict(self, data: List[int]) -> int:
        signal_sum = self.__get_sum_for_weights(data)
        return int(self.__get_activation_function(signal_sum))

    @staticmethod
    def __get_activation_function(signal_sum: float) -> float:
        return 1. if signal_sum > 1 else 0.

    @staticmethod
    def __get_error(target: int, result: float) -> float:
        return float(target) - result

    def __get_sum_for_weights(self, data: List[int]) -> float:
        length = len(data)
        if length != len(self.weights):
            raise Exception
        return float(np.dot(data, self.weights))

    def __normalize_weights(self, error: float, data: List[int], learning_rate: float) -> None:
        for i in range(len(data)):
            self.weights[i] += error * learning_rate * data[i]
