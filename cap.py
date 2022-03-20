# Standard imports
import cv2
import numpy as np

from enum import Enum
import time

# Modelling
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA


# Screenshot tools
from mss import mss
from PIL import Image
from PIL import ImageGrab

# Game classes
from hough import GetOffset
from overlay import Overlay
from classifier import BackgroundModel
from game import Game
from motion import findTranslation
from template import match_all



CAP_SIZE = (0,80,1260,880)
CAP_REDUCED_SIZE = (800,600)

# Grid sizes
X_SIZE = 13
Y_SIZE = 15

TL,TR,BL,BR = ([0, 173],[692,  61],[87, 544],[785, 430])
RESIZE_SIZE = (402,300)
padding = np.zeros((300,1,4))
PREDICTION_RESIZE_SIZE = ( X_SIZE * 30 * 2 , Y_SIZE * 20* 2 )
# 1280 x 800
#TL,TR,BL,BR = ([18, 169],[699,  57],[108, 544],[792, 429])

# Transform
pts1 = np.float32((TL,TR,BL,BR))
pts2 = np.float32([[0,0],[CAP_REDUCED_SIZE[0],0],[0,CAP_REDUCED_SIZE[1]],[CAP_REDUCED_SIZE[0],CAP_REDUCED_SIZE[1]]])
M = cv2.getPerspectiveTransform(pts1,pts2)
M_I = np.linalg.inv(M)

DirectionToValues = {
    "Up" : [0, 1],
    "Down" : [0, -1],
    "Left" : [1, 0],
    "Right" : [-1, 0],
}
x_offset = 0
x_offset_moving_mode = [0] * 5
x_offset_ndx = 0

def moveCorner(corner, direction):
    if corner == "Top Left":
        global TL
        TL = np.add(TL,DirectionToValues[direction])
        print(TL)
    if corner == "Top Right":
        global TR
        TR = np.add(TR,DirectionToValues[direction])
        print(TR)
    if corner == "Bottom Left":
        global BL
        BL = np.add(BL,DirectionToValues[direction])
        print(BL)
    if corner == "Bottom Right":
        global BR
        BR = np.add(BR,DirectionToValues[direction])
        print(BR)



MOST_RECENT_X_FILE = "X.npy"
MOST_RECENT_Y_FILE = "y.npy"

class InputOptions(Enum):
    PRINT = ord('p')
    FIT = ord('f')
    RESET = ord('x')
    SAVE = ord('s')
    LIVE = ord('i')


CornerClasses = {
    ord('1') : "Top Left",
    ord('2') : "Top Right",
    ord('3') : "Bottom Left",
    ord('4') : "Bottom Right",
}

MoveClasses = {
    0 : "Up",
    1 : "Down",
    2 : "Left",
    3 : "Right",
}


ColorClasses = {
    ord('w') : "Water",
    ord('t') : "Trees",
    ord('r') : "Rock",
    ord('b') : "Bedrock",
    ord('g') : "Ground",
    ord('o') : "Player",
    ord('m') : "Monies",
}


ClassToColor = {
    ord('w') : np.array([0,255,255,255]),
    ord('t') : np.array([0,128,0,255]),
    ord('r') : np.array([128,0,0,255]),
    ord('b') : np.array([128,128,128,255]),
    ord('g') : np.array([0,255,0,255]),
    ord('o') : np.array([255,128,0,255]),
    ord('m') : np.array([255,255,0,255]),
}

class SquareData:
    COLOR_SIZE = 3
    SQUARE_SIZE = X_SIZE * Y_SIZE
    SIZE_INCREMENT = 100

    def __init__(self):
        self.size = 0
        self.capacity = self.SIZE_INCREMENT
        self.data_size = self.SQUARE_SIZE * self.COLOR_SIZE

        # Color point (rgb)
        self.X = np.empty((self.capacity, self.data_size))
        # Labels (int) from ClassMap
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


class Segmenter:
    
    square_data = SquareData()
    current_segment = 0
    img = None
    clf = None
    record = True
    pca = None
    background_model = BackgroundModel("plains")
    
    def __init__(self, img):
        self.img = img

    def capture_square(self, x, y, x0, y0):
        x_0 = (x // X_SIZE) * X_SIZE  + x0
        x_1 = x_0 + X_SIZE
        y_0 = (y // Y_SIZE) * Y_SIZE + y0
        y_1 = y_0 + Y_SIZE
        square = self.img[y_0:y_1, x_0:x_1]
        return square.flatten()
    
    def get_capture(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            square = self.capture_square(x,y,x_offset,0)
            print(x,y)
            if self.record:
                self.record_square(square)
            else:
                prediction_xy = self.predict(np.array(square))
                print(self.background_model.classes.get(prediction_xy[0]))


    def flip_mode(self):
        self.record = not self.record
        print("Recording" if self.record else "Predicting")

    def record_square(self, square):
        self.background_model.add_observation(X=square, y=self.current_segment)
    
    def increment_segment(self):
        self.current_segment = (self.current_segment + 1) % self.background_model.num_classes
        self.print_current_segment()   

    def decrement_segment(self):
        self.current_segment = (self.current_segment - 1) % self.background_model.num_classes
        self.print_current_segment()

    def change_segment(self, s):
        self.current_segment = s
        self.print_current_segment()

    def print_current_segment(self):
        print(self.background_model.classes.get(self.current_segment))
    
    def fit_model(self):
        """

        print("Fitting model for %d datapoints" % self.square_data.size)

        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        #self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
        #              hidden_layer_sizes=(15, ), random_state=1)

        X = self.square_data.get_X()
        Y = self.square_data.get_y()

        self.pca = PCA(n_components=10).fit(X) 
        X = self.pca.transform(X)

        self.clf.fit(X / 255, Y)
        """

    def predict(self, square):
        return self.background_model.predict(X=square)

    def predict_img(self):
        y,x,c = self.img.shape

        num_x = x // X_SIZE
        RE_X = num_x * X_SIZE
        y_offset = 0
        num_y = y // Y_SIZE
        RE_Y = num_y * Y_SIZE
        print(num_x,num_y)
  

        X = np.empty((num_x * num_y, self.square_data.data_size))
        for i in range(0, num_x):
            for j in range(y_offset, num_y + y_offset):
                ndx = (j - y_offset)  + i * num_y
                X[ndx] = self.capture_square(i * X_SIZE, j * Y_SIZE, 0, 0)

        print("Predicting for Image")
        return self.predict(X)

    def save(self):
        self.background_model.save()

    def load(self):
        print("Loading data")
        with open(MOST_RECENT_X_FILE, 'rb') as f:
            self.square_data.X = np.load(f)

        with open(MOST_RECENT_Y_FILE, 'rb') as f:
            self.square_data.y = np.load(f)

        self.square_data.size = self.square_data.X.shape[0]
        self.square_data.capacity = self.square_data.size
        self.square_data.resize_if_full()


def drawLines(img, x,y, offset_x, offset_y):
    # Black color in BGR 
    color = (0, 0, 0) 
      
    # Line thickness of 1 px 
    thickness = 1

    r,c, _ = img.shape
    for y_i in range(offset_y, r,y):
        start = (0, y_i)
        end = (c, y_i)
        img = cv2.line(img, start, end, color, thickness)
    for x_i in range(offset_x, c,x):
        start = (x_i, 0)
        end = (x_i, r)
        img = cv2.line(img, start, end, color, thickness)

    return img


def RenderBoard(img):
    img = np.transpose(img, (1, 0, 2))
    img = cv2.resize(img, dsize=PREDICTION_RESIZE_SIZE, interpolation=cv2.INTER_NEAREST)
    img = drawLines(img, X_SIZE * 2, Y_SIZE  * 2, 0, 0)
    img = cv2.warpPerspective(img,M_I,CAP_REDUCED_SIZE)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=CAP_SIZE, interpolation=cv2.INTER_NEAREST)

def most_common(lst):
    return max(set(lst), key=lst.count)



def capture_screenshot():
    # Capture entire screen
    with mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        # Convert to PIL/Pillow Image
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')


def GrabScreenshot():
    t0 = time.time()
    img = capture_screenshot() #ImageGrab.grab(bbox=CAP_SIZE) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img = img.crop(CAP_SIZE)
    t1 = time.time()
    print("Capture time - ", t1 - t0)
    img_np = np.array(img) #this is the array obtained from conversion
    img_np = cv2.resize(img_np, dsize=CAP_REDUCED_SIZE, interpolation=cv2.INTER_CUBIC)


    # Warp the captured image
    #                   TL       TR       BL        BR
    img_np = cv2.warpPerspective(img_np,M,CAP_REDUCED_SIZE)

    img_np = cv2.resize(img_np, dsize=RESIZE_SIZE, interpolation=cv2.INTER_CUBIC) 

    t2 = time.time()
    print("Warp time - ", t2 - t1)


    """
    # 13,11
    # 300 / 15 ~= 27, 400 / 13 ~= 30
    global x_offset
    global x_offset_moving_mode
    global x_offset_ndx
    x_offset_moving_mode[x_offset_ndx] = int(round(GetOffset(img_np)))
    x_offset_ndx = (x_offset_ndx + 1) % 5
    x = most_common(x_offset_moving_mode)
    t3 = time.time()
    print("Offset time - ", t3 - t2)
    img_np = np.roll(img_np, -x, axis=1)
    t4 = time.time()
    print("Roll time - ", t4 - t3)
    """
    print(GetOffset(img_np))
    t3 = time.time()
    print("Offset time - ", t3 - t2)

    return img_np

img_np = GrabScreenshot()
curr_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
cv2.namedWindow("original")
cv2.moveWindow("original", 800, 0)
cv2.namedWindow("prediction")
cv2.moveWindow("prediction", 1200, 0)
cv2.namedWindow("board")
cv2.moveWindow("board", 1600, 0)
cv2.imshow("original", frame)
cv2.imshow("prediction", frame)
cv2.imshow("board", frame)

game = Game()
distance = 0

"""
cv2.namedWindow("tracking")
bbox = cv2.selectROI("tracking", frame)
tracker = cv2.TrackerMIL_create()
ok = tracker.init(frame, bbox)
"""

seg = Segmenter(img_np)
cv2.setMouseCallback("original", seg.get_capture)
live_prediction = False
current_state = "Paused"
sub_state = "Started"
while(1):
    k = cv2.waitKey(200) & 0xFF

    if current_state == "Live Prediction":
        t0 = time.time()

        prev_gray = curr_gray
        img_np = GrabScreenshot()

        # Find motion
        curr_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        diff = findTranslation(prev_gray, curr_gray)
        print("Translation = ", diff)
        distance += diff
        img_np = np.roll(img_np, -int(distance) % X_SIZE, axis=1)


        seg.img = img_np

        # Predict current state
        t1 = time.time()
        prediction, prediction_img = seg.predict_img()
        t2 = time.time()
        # Update game board
        game.board.add_prediction(prediction, 0)
        board_rgb = game.board.get_board_rgb( 0)
        # Draw current prediction and game board
        cv2.imshow("prediction", RenderBoard(prediction_img))
        cv2.imshow("board", RenderBoard(board_rgb))
        # Show current capture
        frame_orig = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        match_all(frame_orig)
        frame_orig = drawLines(frame_orig, X_SIZE, Y_SIZE, x_offset, 0)
        frame_orig = Overlay(frame_orig, current_state)
        cv2.imshow("original", frame_orig)
        t3 = time.time()
        print("Total capture time: %s" % (t1 - t0))
        print("Total prediction time: %s" % (t2 - t1))
        print("Total render time: %s" % (t3 - t2))

        """
        ok, newbox = tracker.update(frame_orig)
        print(ok, newbox)

        if ok:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame_orig, p1, p2, (200,0,0))
        cv2.imshow("tracking", frame_orig)
        """


    elif current_state == "Moving Corners":
        img_np = GrabScreenshot()
        frame_orig = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        frame_orig = Overlay(frame_orig, current_state + " : " + sub_state)
        cv2.imshow("original", frame_orig)
    else:
        frame_orig = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        frame_orig = Overlay(frame_orig, current_state)
        cv2.imshow("original", frame_orig)



    if k == 27:
        break
    elif k == ord('p'):
        # Predict current state
        t0 = time.time()
        prediction, prediction_img = seg.predict_img()
        t1 = time.time()
        # Update game board
        game.board.add_prediction(prediction, 0)
        board_rgb = game.board.get_board_rgb( 0)
        # Draw current prediction and game board
        cv2.imshow("prediction", RenderBoard(prediction_img))
        cv2.imshow("board", RenderBoard(board_rgb))
        t2 = time.time()
        print("Total prediction time: %s" % (t1 - t0))
        print("Total render time: %s" % (t2 - t1))
    elif k == ord('f'):
        seg.fit_model()
    elif k == ord('x'):
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        cv2.imshow("prediction", frame)
        current_state = "Paused"
    elif k == ord('s'):
        seg.save()
    elif k == ord('l'):
        seg.load()
    elif k == ord('i'):
        current_state = "Live Prediction"
    elif k == ord('['):
        seg.increment_segment()
    elif k == ord(']'):
        seg.decrement_segment()
    elif k in CornerClasses:
        current_state = "Moving Corners"
        sub_state = CornerClasses.get(k)
    elif k in MoveClasses and current_state == "Moving Corners":
        moveCorner(sub_state, MoveClasses[k])
    elif k in ColorClasses:
        seg.change_segment(k)
        current_state = ColorClasses.get(k)





