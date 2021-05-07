import cv2
import numpy as np
from  collections import Counter

dilate_kernel = np.ones((5,5),np.uint8)
erode_kernel = np.ones((3,3),np.uint8)

vert_lower = 0.1 / 180 * np.pi
vert_upper = 179.9 / 180 * np.pi

def is_vertical(line):
    rho,theta = line[0]
    return theta < vert_lower or theta > vert_upper

hznt_lower = 89.9 / 180 * np.pi
hznt_upper = 90.1 / 180 * np.pi

def is_horizontal(line):
    rho,theta = line[0]
    return theta > hznt_lower and theta < hznt_upper


def avg_offset(offsets):
    sum1 = 0
    sum2 = 0
    size = 0
    for k,v in offsets:
        size += v
        if k > 6.5:
            sum1 += (k - 13) * v
        else:
            sum1 += k * v
        sum2 += k * v

    if size == 0:
        return (0,0)

    return (sum1 / size, sum2 / size)


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

def GetOffset(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    edges = cv2.dilate(edges,dilate_kernel,iterations = 1)
    edges = cv2.erode(edges,erode_kernel,iterations = 1)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
    if len(lines) == 0:
        return 0

    x_offset = Counter()
    vert_lines = [l  for l in lines if is_vertical(l)]
    if len(vert_lines) == 0:
        return 0

    for line in vert_lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        x1 = int(x0 + 1000*(-b))

        offset = x1 % 13
        x_offset[offset] += 1
        
        img = cv2.line(img, start, end, color, thickness)

    return avg_offset(x_offset.most_common(3))[0]
