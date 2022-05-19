import json
import numpy as np

def load_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    glob_res_x = []
    y_res = []
    for key, value in data.items():
        size = len(value)
        for subval in value:
            result = []
            for i in range(int(len(subval) / size)):
                result.append(subval[i * size: (i + 1) * size])
            glob_res_x.append(result)
        y_res += [int(key)] * size

    return np.array(glob_res_x), np.array(y_res)


print(load_json('train.json'))
