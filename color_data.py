


class ColorData:
    LABEL_SIZE = 1
    COLOR_SIZE = 4
    SIZE_INCREMENT = 100
    data_size = 0

    def __init__(self, data_size):
        self.size = 0
        self.capacity = self.SIZE_INCREMENT
        self.data_size = data_size * self.COLOR_SIZE

        # Color point (rgb)
        self.X = np.empty((self.capacity, self.data_size))
        # Labels (int) from ColorClasses
        self.y = np.empty((self.capacity))
    
    def get_X(self):
        return self.X[0:self.size][:]

    def get_y(self):
        return self.y[0:self.size][:]

    def resize_if_full(self):
        if self.size >= self.capacity - 1:
            print("Increasing Data Capacity")
            self.capacity += self.SIZE_INCREMENT
            self.X.resize((self.capacity, self.data_size))
            self.y.resize((self.capacity))
    
    def append(self, X, y):
        self.resize_if_full()
        self.X[self.size] = X
        self.y[self.size] = y
        self.size += 1