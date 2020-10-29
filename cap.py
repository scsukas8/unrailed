from PIL import ImageGrab
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import cv2



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
        #self.X = np.empty((self.capacity,self.COLOR_SIZE))
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
    
    window_size = 2
    color_data = ColorData((window_size * 2) ** 2)
    current_segment = 0
    current_color = None
    img = None
    clf = None
    record = True
    
    def __init__(self, img):
        self.img = img

    def capture_color(self, x, y):
        xl = x - self.window_size
        xr = x + self.window_size
        yu = y - self.window_size
        yd = y + self.window_size
        window = self.img[xl:xr,yu:yd]
        return window.flatten()
    
    def get_capture(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_color = self.capture_color(y,x)
            print(x,y)
            if self.record:
                self.record_data()
            else:
                prediction_xy = self.predict(np.array([self.current_color]))
                print(ColorClasses.get(prediction_xy[0]))


    def flip_mode(self):
        self.record = not self.record
        print("Recording" if self.record else "Predicting")

    def record_data(self):
        self.color_data.append(X=self.current_color, y=self.current_segment)
        self.print_current_segment()

    
    def change_segment(self, s):
        self.current_segment = s
        self.print_current_segment()

    def print_current_segment(self):
        print(ColorClasses.get(self.current_segment))

    def print(self):
        print(self.color_data.get_X())
        print(self.color_data.get_y())
    
    def fit_model(self):
        print("Fitting model for %d datapoints" % self.color_data.size)
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        self.clf.fit(self.color_data.get_X(), self.color_data.get_y())
    

    def predict(self, color):
        Y = self.clf.predict(color)
        return Y

    def predict_img(self):

        X = np.empty((self.img.shape[0] * self.img.shape[1], self.color_data.data_size))

        for x in range(self.window_size, self.img.shape[0] - self.window_size - 1):
            for y in range(self.window_size, self.img.shape[1] - self.window_size - 1):
                ndx = x + y * self.img.shape[0]
                X[ndx] = self.capture_color(x,y)

        print("Predicting for Image")
        Y = self.predict(X)
        print("Converting to Color")
        prediction_img = np.zeros(self.img.shape, dtype=np.uint8)
        for x in range(self.window_size, self.img.shape[0] - self.window_size - 1):
            for y in range(self.window_size, self.img.shape[1] - self.window_size - 1):
                ndx = x + y * self.img.shape[0]
                prediction_img[x,y] = ClassToColor.get(Y[ndx])

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




img = ImageGrab.grab(bbox=(0,80,800,680)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
img_np = np.array(img) #this is the array obtained from conversion

# Warp the captured image
rows,cols, ch = img_np.shape
print(rows,cols)
#                   TL       TR       BL        BR
pts1 = np.float32([[10,140],[685,43],[119,567],[805,467]])
pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
M = cv2.getPerspectiveTransform(pts1,pts2)
img_np = cv2.warpPerspective(img_np,M,(rows,cols))


#
print(img_np.shape)
img_np = cv2.resize(img_np, dsize=(400,300), interpolation=cv2.INTER_CUBIC)

# 13,11
# 600 / 11 ~= 27, 800 / 13 ~= 61
img_np = drawLines(img_np, 13, 11, 0, 0)


frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
cv2.imshow("test", frame)


seg = Segmenter(img_np)
cv2.namedWindow("test")
cv2.setMouseCallback("test", seg.get_capture)

while(1):
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
    elif k == ord('p'):
        prediction = seg.predict_img()
        alpha = 0.5
        beta = (1.0 - alpha)
        print("Blending Images")
        img_blend = cv2.addWeighted(img_np, alpha, prediction, beta, 0.0)
        frame = cv2.cvtColor(img_blend, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", frame)
    elif k == ord('f'):
        seg.fit_model()
    elif k == ord('m'):
        seg.flip_mode()
    else:
        seg.change_segment(k)




