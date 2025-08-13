from ultralytics import YOLO
import time
import random
import math
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

def draw_fire_effect(img, num_people, frame_count=0):
    """
    Draw animated fire effect using OpenCV drawing functions
    """
    if num_people == 0:
        return img
    
    height, width = img.shape[:2]
    
    # Fire parameters based on number of people
    base_flame_height = min(50 + (num_people * 30), height // 3)
    num_flames = min(3 + num_people * 2, 15)
    
    # Fire colors (BGR format) - from yellow to red
    fire_colors = [
        (0, 255, 255),    # Yellow
        (0, 200, 255),    # Orange-yellow
        (0, 150, 255),    # Orange
        (0, 100, 255),    # Red-orange
        (0, 50, 200),     # Red
        (0, 0, 150)       # Dark red
    ]
    
    # Draw flames from bottom of screen
    for i in range(num_flames):
        # Random flame position along bottom
        flame_x = int((i + 1) * width / (num_flames + 1) + random.randint(-20, 20))
        flame_base_y = height - 10
        
        # Animated flame height with some randomness
        flame_height = base_flame_height + int(20 * math.sin(frame_count * 0.1 + i)) + random.randint(-10, 10)
        flame_top_y = max(flame_base_y - flame_height, 50)
        
        # Draw flame as multiple overlapping ellipses
        for j in range(5):
            # Flame width decreases as we go up
            width_factor = (5 - j) / 5.0
            flame_width = int(15 * width_factor * (1 + num_people * 0.3))
            
            # Y position for this flame segment
            segment_y = int(flame_base_y - (j * flame_height / 5))
            
            # Add some wobble animation
            wobble_x = int(10 * math.sin(frame_count * 0.2 + i + j) * width_factor)
            
            # Color gets redder towards the base
            color_index = min(j, len(fire_colors) - 1)
            color = fire_colors[color_index]
            
            # Draw ellipse for flame segment
            cv2.ellipse(img, 
                       (flame_x + wobble_x, segment_y), 
                       (flame_width, int(flame_height / 6)), 
                       0, 0, 180, color, -1)
            
            # Add inner brighter core
            if j < 3:
                cv2.ellipse(img, 
                           (flame_x + wobble_x, segment_y), 
                           (max(1, flame_width // 3), max(1, int(flame_height / 8))), 
                           0, 0, 180, (100, 255, 255), -1)
    
    # Add sparks/embers
    for _ in range(num_people * 3):
        spark_x = random.randint(50, width - 50)
        spark_y = random.randint(height - base_flame_height - 50, height - 20)
        spark_size = random.randint(1, 3)
        cv2.circle(img, (spark_x, spark_y), spark_size, (0, 150, 255), -1)
    
    return img

def draw_sidebar(img, num_people, fire_duration):
    """
    Draw sidebar with people count and fire duration
    """
    height, width = img.shape[:2]
    sidebar_width = 200
    
    # Draw sidebar background
    cv2.rectangle(img, (width - sidebar_width, 0), (width, height), (40, 40, 40), -1)
    cv2.rectangle(img, (width - sidebar_width, 0), (width, height), (80, 80, 80), 2)
    
    # Title
    cv2.putText(img, "FIRE KEEPER", (width - sidebar_width + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # People count
    cv2.putText(img, "PEOPLE:", (width - sidebar_width + 10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, str(num_people), (width - sidebar_width + 10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    
    # Fire status
    if fire_duration > 0:
        cv2.putText(img, "FIRE ACTIVE:", (width - sidebar_width + 10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Format duration as MM:SS
        minutes = int(fire_duration // 60)
        seconds = int(fire_duration % 60)
        duration_text = f"{minutes:02d}:{seconds:02d}"
        cv2.putText(img, duration_text, (width - sidebar_width + 10, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Fire intensity bar
        bar_height = min(int(fire_duration * 3), 100)
        cv2.rectangle(img, (width - 50, height - 150), (width - 20, height - 150 + bar_height), 
                     (0, 165, 255), -1)
        cv2.rectangle(img, (width - 50, height - 150), (width - 20, height - 50), 
                     (255, 255, 255), 1)
        cv2.putText(img, "HEAT", (width - 55, height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    else:
        cv2.putText(img, "FIRE:", (width - sidebar_width + 10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "DORMANT", (width - sidebar_width + 10, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    return img

def run_demo():
    # USB camera on WSL2
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)
    #    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # Create a mask of logo

    assert cap.isOpened(), "Camera not found. Please check the camera connection."
    frame_count = 0
    fire_start_time = None
    fire_duration = 0
    while True:
        _, img = cap.read()
        frame_count += 1
        
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
        # Fire timer logic
        current_time = time.time()
        if result_len > 0:
            # People detected - start or continue fire
            if fire_start_time is None:
                fire_start_time = current_time
            fire_duration = current_time - fire_start_time
        else:
            # No people detected - extinguish fire
            fire_start_time = None
            fire_duration = 0
        img = draw_fire_effect(img, result_len, frame_count)
        img = draw_sidebar(img, result_len, fire_duration)

        cv2.imshow('YOLO 11 Detection', img)    
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()

list_ports()

run_demo()
