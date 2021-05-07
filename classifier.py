import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import cv2
import collections
import matplotlib.colors


Class = collections.namedtuple('Class', 'name color ')

def color_to_rgb(color):
    c_tuple = matplotlib.colors.to_rgb( matplotlib.colors.CSS4_COLORS[color])
    c_list = [x * 255 for x in c_tuple]
    return c_list


from model import Model




class BackgroundModel(Model):
    # The background is 30 x 20, where each element is 13 x 15 x 3
    X_SIZE = 30
    Y_SIZE = 20
    X_DATA_SIZE = 13
    Y_DATA_SIZE = 15
    CHANNEL_SIZE = 3

    def __init__(self, name):
        self.num_classes = 0        
        self.classes = {}
        self._setup_classes()
        super().__init__(name, self.X_DATA_SIZE * self.Y_DATA_SIZE * self.CHANNEL_SIZE)


    def _add_class(self, name, color):
        self.classes[self.num_classes] = Class(name, color)
        self.num_classes += 1

    def _setup_classes(self):
      self._add_class(name="None", color="black")
      self._add_class(name="Water", color="aqua")
      self._add_class(name="Trees", color="green")
      self._add_class(name="Rocks", color="sienna")
      self._add_class(name="Bedrock", color="grey")
      self._add_class(name="Ground", color="greenyellow")
      self._add_class(name="Ground_Bridge", color="greenyellow")
      self._add_class(name="Ground_Rock", color="greenyellow")
      self._add_class(name= "Ground_Tree", color="greenyellow")
      self._add_class(name= "Player", color="yellow")
      self._add_class(name= "Bolt", color="gold")
      self._add_class(name= "Tracks", color="rosybrown")
      self._add_class(name= "Train", color="silver")
      self._add_class(name= "Station", color="teal")
      self._add_class(name= "WaitingRoom", color="blue")

    def _prediction_to_rgb(self, y):
        return color_to_rgb(self.classes[y].color)


    def predict(self, X):
        y = super().predict(X)
        y_rgb = np.array([self._prediction_to_rgb(yi) for yi in y], dtype=np.uint8)
        prediction = y.reshape((self.X_SIZE,self.Y_SIZE))
        prediction_rgb = y_rgb.reshape((self.X_SIZE,self.Y_SIZE, self.CHANNEL_SIZE))
        return prediction, prediction_rgb

    def add_observation(self, X, y):
        super().add_observation(X, y)





