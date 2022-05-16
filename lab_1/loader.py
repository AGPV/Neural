import json
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class TrainData:
    target: int
    number: int
    data: List[int]


class DataLoader:
    def __init__(self, target: int = 0) -> None:
        self.data_path = 'data.json'
        if 0 <= target <= 9:
            self.target = str(target)
        else:
            raise AttributeError
        self.train_data: Dict[str, list] = {}

    def load_data(self) -> list[TrainData]:
        with open(self.data_path) as f:
            self.train_data = json.load(f)
        res_array = []
        for item in self.train_data:
            target_num = 0
            if self.target == item:
                target_num = 1
            res_array.append(TrainData(target=target_num, number=int(item), data=self.train_data[item]))
        return res_array
