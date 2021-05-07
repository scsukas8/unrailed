import cv2
import numpy as np
import matplotlib.pyplot as plt

template_dir = "templates/"
template_ext = ".png"
template_names = ["bolt", "bolt_2", "bolt_3", "double_bolt", "station",
                  "station_desert", "axe_small", "bucket_small", "pickaxe_small"]
templates = {t: cv2.imread(
    template_dir + t + template_ext, cv2.IMREAD_COLOR
) for t in template_names}

ref_img = cv2.imread('templates/example.png', cv2.IMREAD_COLOR)

img = None
with open("example_imp.npy", 'rb') as f:
    img = np.load(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


w, h = img.shape[:2]
ref_w, ref_h = ref_img.shape[:2]
ratio = (w / ref_w, h / ref_h)
print(ratio)
print(w, h, ref_h, ref_h)

templates = {t: cv2.resize(t_img, (0, 0), fx=ratio[0], fy=ratio[1])
             for t, t_img in templates.items()}


def match_template(img, template):
    # Apply template Matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    threshold = 0.75

    return max_val > threshold,  max_loc


def match_all(img):

    for t, t_img in templates.items():
        match, top_left = match_template(img, t_img)
        if not match:
            continue

        _, w, h = t_img.shape[::-1]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)


if __name__ == "__main__":
    match_all(img)
    cv2.imshow("res", img)
    cv2.waitKey(0)
