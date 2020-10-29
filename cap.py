from PIL import ImageGrab
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import cv2

# Grid sizes
X_SIZE = 13
Y_SIZE = 11

ColorClasses = {
    ord('w') : "Water",
    ord('t') : "Trees",
    ord('r') : "Rock",
    ord('b') : "Bedrock",
    ord('g') : "Ground",
    ord('o') : "Orange"
}


ClassToColor = {
    ord('w') : np.array([0,255,255,255]),
    ord('t') : np.array([0,128,0,255]),
    ord('r') : np.array([128,0,0,255]),
    ord('b') : np.array([128,128,128,255]),
    ord('g') : np.array([0,255,0,255]),
    ord('o') : np.array([255,128,0,255])
}

class SquareData:
    COLOR_SIZE = 4
    SQUARE_SIZE = X_SIZE * Y_SIZE
    SIZE_INCREMENT = 100

    def __init__(self):
        self.size = 0
        self.capacity = self.SIZE_INCREMENT
        self.data_size = self.SQUARE_SIZE * self.COLOR_SIZE

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


class Segmenter:
    
    square_data = SquareData()
    current_segment = 0
    img = None
    clf = None
    record = True
    
    def __init__(self, img):
        self.img = img

    def capture_square(self, x, y):
        x_0 = (x // X_SIZE) * X_SIZE
        x_1 = x_0 + X_SIZE
        y_0 = (y // Y_SIZE) * Y_SIZE
        y_1 = y_0 + Y_SIZE
        square = self.img[y_0:y_1, x_0:x_1]
        return square.flatten()
    
    def get_capture(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            square = self.capture_square(x,y)
            print(x,y)
            if self.record:
                self.record_square(square)
            else:
                prediction_xy = self.predict(np.array(square))
                print(ColorClasses.get(prediction_xy[0]))


    def flip_mode(self):
        self.record = not self.record
        print("Recording" if self.record else "Predicting")

    def record_square(self, square):
        self.square_data.append(X=square, y=self.current_segment)
    
    def change_segment(self, s):
        self.current_segment = s
        self.print_current_segment()

    def print_current_segment(self):
        print(ColorClasses.get(self.current_segment))

    def print(self):
        print(self.square_data.get_X())
        print(self.square_data.get_y())
    
    def fit_model(self):
        print("Fitting model for %d datapoints" % self.square_data.size)
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        self.clf.fit(self.square_data.get_X(), self.square_data.get_y())

    def predict(self, square):
        Y = self.clf.predict(square)
        return Y

    def predict_img(self):
        y,x,c = self.img.shape

        num_x = x // X_SIZE
        y_offset = 3
        y_top_offset = 4
        num_y = y // Y_SIZE
        num_y = num_y - y_offset - y_top_offset # remove top 3 and bottom 4
  

        X = np.empty((num_x * num_y, self.square_data.data_size))
        for i in range(0, num_x):
            for j in range(y_offset, num_y + y_offset):
                ndx = (j - y_offset)  + i * num_y
                X[ndx] = self.capture_square(i * X_SIZE, j * Y_SIZE)

        print("Predicting for Image")
        Y = self.predict(X)
        print("Converting to Color")


        prediction_img = np.zeros(( y // Y_SIZE,num_x , c), dtype=np.uint8)
        for i in range(0, num_x):
            for j in range(y_offset, num_y + y_offset):
                ndx = (j - y_offset)  + i * num_y
                prediction_img[j,i ] = ClassToColor.get(Y[ndx])

        prediction_img = cv2.resize(prediction_img, dsize=(400,300), interpolation=None)

        return prediction_img

def drawLines(img, x,y, offset_x, offset_y):
    # Black color in BGR 
    color = (0, 0, 0) 
      
    # Line thickness of 1 px 
    thickness = 1

    r,c, _ = img.shape
    for y_i in range(offset_y, r,y):
        start = (0, y_i)
        end = (c, y_i)
        image = cv2.line(img, start, end, color, thickness)
    for x_i in range(offset_x, c,x):
        start = (x_i, 0)
        end = (x_i, r)
        image = cv2.line(img, start, end, color, thickness)

    return image


def GrabScreenshot():

    img = ImageGrab.grab(bbox=(0,80,800,680)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img_np = np.array(img) #this is the array obtained from conversion

    # Warp the captured image
    rows,cols, ch = img_np.shape
    #                   TL       TR       BL        BR
    pts1 = np.float32([[10,140],[685,43],[119,567],[805,467]])
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    img_np = cv2.warpPerspective(img_np,M,(rows,cols))


    img_np = cv2.resize(img_np, dsize=(400,300), interpolation=cv2.INTER_CUBIC)

    # 13,11
    # 300 / 11 ~= 27, 400 / 13 ~= 30
    img_np = drawLines(img_np, X_SIZE, Y_SIZE, 0, 0)
    return img_np

img_np = GrabScreenshot()
frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
cv2.imshow("test", frame)


seg = Segmenter(img_np)
cv2.namedWindow("test")
cv2.setMouseCallback("test", seg.get_capture)
live_prediction = False
while(1):
    k = cv2.waitKey(200) & 0xFF

    if live_prediction:
        img_np = GrabScreenshot()
        seg.img = img_np
        seg.fit_model()
        prediction = seg.predict_img()
        alpha = 0.1
        beta = (1.0 - alpha)
        print("Blending Images")
        img_blend = cv2.addWeighted(img_np, alpha, prediction, beta, 0.0)
        frame = cv2.cvtColor(img_blend, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", frame)


    if k == 27:
        break
    elif k == ord('p'):
        prediction = seg.predict_img()
        alpha = 0.1
        beta = (1.0 - alpha)
        print("Blending Images")
        img_blend = cv2.addWeighted(img_np, alpha, prediction, beta, 0.0)
        frame = cv2.cvtColor(img_blend, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", frame)
    elif k == ord('f'):
        seg.fit_model()
    elif k == ord('m'):
        seg.flip_mode()
    elif k == ord('x'):
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", frame)
        live_prediction = False
    elif k == ord('l'):
        live_prediction = not live_prediction
    elif k in ColorClasses:
        seg.change_segment(k)




