import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


class Model:

    def __init__(self, name, data_size):
        self._name = name

        self.clf = None
        self.pca = None
        self.data = ModelData(name, data_size)
        self.new_observations = False

        # Load inital data and fit if requested.
        self.data.load()
        self._fit()


    def _fit(self):
        if self.data.size == 0:
            print("Model " + self._name + " has no data, skipping. ")
            return
        print("Fitting model " + self._name + " for %d datapoints" % self.data.size)
        
        X = self.data.X
        y = self.data.y

        #self.pca = PCA(n_components=10).fit(X) 
        #X = self.pca.transform(X)

        pipe = make_pipeline( StandardScaler(),
                            PCA(n_components=10),
                            SVC(gamma='auto'))


        param_grid = dict(pca__n_components=[8,10,12],
                          svc__C=np.logspace(-4, 1, 6),
                          svc__kernel=['rbf','linear'])

        self.clf = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1, verbose=2)

        self.clf.fit(X / 255, y)

    def predict(self, X):
        if self.new_observations:
            self._fit()
            self.new_observations = False

        #X = self.pca.transform(X)
        y = self.clf.predict(X / 255)
        return y

    def add_observation(self, X, y):
        self.data.append(X, y)
        self.new_observations = True

    def save(self):
        self.data.save()





class ModelData:
    # Amount to scale storage by when full.
    SIZE_INCREMENT = 100
    MODEL_FOLDER = "models/"
    MODEL_EXT = ".npy"

    def __init__(self, name, data_size):
        # Size variables
        self._size = 0
        self._capacity = self.SIZE_INCREMENT
        self._data_size = data_size

        # Input - Color point (rgb)
        self._X = np.empty((self._capacity, self._data_size))
        # Predictions - Labels (int) from ClassMap
        self._y = np.empty((self._capacity))

        # File management
        self._name = name
        self._save_file_X = self.MODEL_FOLDER + name + "_X" + self. MODEL_EXT
        self._save_file_y = self.MODEL_FOLDER + name + "_y" + self. MODEL_EXT

    @property
    def size(self):
        return self._size

    @property
    def X(self):
        return self._X[0:self._size][:]

    @property
    def y(self):
        return self._y[0:self._size][:]

    def _resize_if_full(self):
        if self._size >= self._capacity - 1:
            print("Increasing Data Capacity")
            self._capacity += self.SIZE_INCREMENT
            self._X.resize((self._capacity, self._data_size))
            self._y.resize((self._capacity))
    
    def append(self, X, y):
        self._resize_if_full()
        self._X[self._size] = X
        self._y[self._size] = y
        self._size += 1

    def save(self):
        print("Saving data for " + self._name)
        with open(self._save_file_X, 'wb') as f:
            np.save(f, self.X)

        with open(self._save_file_y, 'wb') as f:
            np.save(f, self.y)


    def load(self):
        print("Loading data for " + self._name)
        try:
            with open(self._save_file_X, 'rb') as f:
                self._X = np.load(f)

            with open(self._save_file_y, 'rb') as f:
                self._y = np.load(f)

        except FileNotFoundError  as e:
            print("No model found. Initializing empty data.")
            return

        self._size = self._X.shape[0]
        self._capacity = self._size
        self._resize_if_full()
