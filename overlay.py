import cv2

def Overlay(img, text):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
      
    # org 
    org = (50, 50) 
      
    # fontScale 
    fontScale = 0.5
       
    # Blue color in BGR 
    color = (255, 0, 0) 
      
    # Line thickness of 2 px 
    thickness = 2
       
    # Using cv2.putText() method 
    img = cv2.putText(img, text, org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
    return img