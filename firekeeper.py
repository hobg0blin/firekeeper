from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator




def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not read." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

def run_demo():
    # USB camera on WSL2
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)
    fire = cv2.imread('fire.png')
    #    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # Create a mask of logo

    assert cap.isOpened(), "Camera not found. Please check the camera connection."

    while True:
        _, img = cap.read()
        
        # BGR to RGB conversion is performed under the hood
        # see: https://github.com/ultralytics/ultralytics/issues/2575
        results = model.predict(img, classes=0)
        result_len = 0
        for r in results:
            
            annotator = Annotator(img)
            
            boxes = r.boxes
            for box in boxes:
                result_len +=1
                
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
        size = 10 * result_len
        if size > 0:
            fire = cv2.resize(fire, (size, size))
            img2gray = cv2.cvtColor(fire, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

            img = annotator.result()  

            # Region of Image (ROI), where we want to insert logo
            roi = img[-size-10:-10, -size-10:-10]

            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += fire
            # Create a mask of logo
            img2gray = cv2.cvtColor(fire, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

              
        cv2.imshow('YOLO 11 Detection', img)     
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

list_ports()

run_demo()
