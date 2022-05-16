from typing import List
from loader import TrainData
from perceptron import Neuron


class Network:
    def __init__(self, train_data: List[TrainData]) -> None:
        num_of_neuron_weights = len(train_data[0].data)
        self.data_length = len(train_data)
        self.train_data = train_data
        self.neurons = [Neuron(num_of_neuron_weights) for _ in range(self.data_length)]

    def train(self, learning_rate: float = 0.1, epochs: int = 10) -> None:
        for i in range(self.data_length * epochs):
            item = self.train_data[i % self.data_length]
            for its_number, neuron in enumerate(self.neurons):
                if item.number == its_number:
                    item.target = 1
                neuron.train(item, learning_rate)
                item.target = 0

    def predict(self, data: List[int]) -> int:
        outputs = [neuron.predict(data=data) for neuron in self.neurons]
        return outputs.index(1)
