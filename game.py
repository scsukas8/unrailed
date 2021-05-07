
import numpy as np
from scipy import stats
from classifier import BackgroundModel
np.core.arrayprint._line_width = 150

model = BackgroundModel("blah")


class Board:

    def __init__(self):
        self.game_window_height = 20
        self.game_window_length = 30
        self.game_length = 100
        self.num_predictions = 10

        # The board is a 20 x L x P size array,where L is the current game distance
        # and P is the number of predictions for a given square.
        self.board = np.zeros(
            (self.game_length, self.game_window_height, self.num_predictions), dtype=np.uint8)

        # Temp tracks index for the prediction
        self.ndx = 0

    # Adds a prediction to the game board. The prediction is one window starting at distance x on the board.

    def add_prediction(self, prediction, x):
        self.board[x:x+self.game_window_length, :, self.ndx] = prediction
        self.ndx += 1
        self.ndx %= self.num_predictions

    def get_board(self, x):
        board = stats.mode(
            self.board[x:x+self.game_window_length, :, :], axis=2)[0]
        return board.reshape(self.game_window_length, self.game_window_height)

    def get_board_rgb(self, x):
        board = self.get_board(x)
        # Flatten the board (20,30) and convert each element to rgb. Then reform the array (20,30,3)
        board_rgb = np.array(
            list(map(model._prediction_to_rgb, board.flatten())), dtype=np.uint8)
        return board_rgb.reshape(self.game_window_length, self.game_window_height, 3)


class Game:

    def __init__(self):
        self.board = Board()
        self.path_finder = None
