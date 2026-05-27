import numpy as np


class Table:
    def __init__(self, name="", data=np.empty((0, 2), dtype=float)) -> None:
        self.name = name
        self.data = data

    def add(self, x: float, y: float) -> None:
        self.data = np.append(self.data, np.array([[x, y]]), axis=0)

    def set_data(self, x: float, y: float) -> None:
        self.add(x, y)

    def get_val(self, x: float) -> float:
        if len(self.data) <= 0:
            return 0.0
        if x <= self.data[0, 0]:
            return float(self.data[0, 1])
        if x >= self.data[-1, 0]:
            return float(self.data[-1, 1])

        for i in range(len(self.data[:, 0])):
            if x <= self.data[i, 0]:
                return float(self.data[i, 1] -
                        (self.data[i, 1] - self.data[i - 1, 1]) * (self.data[i, 0] - x) / (self.data[i, 0] - self.data[i - 1, 0]))


    def get_der_val(self, x, y) -> float:
        if len(self.data) <= 0:
            return 0.0
        if x <= self.data[0, 0]:
            return 0.0
        if x >= self.data[-1, 0]:
            return 0.0

        for i in range(len(self.data[:, 0])):
            if x <= self.data[i, 0]:
                return float((self.data[i, 1] - self.data[i - 1, 1])  / (self.data[i, 0] - self.data[i - 1, 0]))

