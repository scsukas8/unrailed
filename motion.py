import cv2
import numpy as np


def findTranslation(prev_gray, curr_gray):

    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)

    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None)

    # Filter out invalid points
    idx = np.where(status == 1)[0]
    motion_diff = curr_pts[idx, 0] - prev_pts[idx, 0]

    # Motion is always horizontal, drop any false positves with vertical motion
    idx = np.where(motion_diff[:, 1] < 0.5)
    horizontal_diff = motion_diff[idx, 0][0]

    # Drop any outliers from the data
    mean = np.mean(horizontal_diff)
    epsilon = 0.001
    sigma = 2
    std = np.std(horizontal_diff)
    idx = np.where(abs(horizontal_diff - mean) < (sigma * std + epsilon))
    horizontal_diff = horizontal_diff[idx]

    # If fewer than 5 points remain, assume no motion.
    if len(horizontal_diff < 5):
        return 0

    return np.average(horizontal_diff)
