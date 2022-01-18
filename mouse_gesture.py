
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import pyautogui as p
import keyboard
import tkinter as tk
from pynput.mouse import Button, Controller

mouse = Controller()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
root=tk.Tk()
global screenRes
screenRes = (root.winfo_screenwidth(),
                 root.winfo_screenheight())  

Val4 = tk.IntVar()
Val4.set(30)
kando = Val4.get()/10
fingerTipIds = [4, 8, 12, 16, 20]
cap_device=0 
def draw_circle(image, x, y, roudness, color):
    cv.circle(image, (int(x), int(y)), roudness, color,
               thickness=5, lineType=cv.LINE_8, shift=0)

def calculate_distance(l1, l2):
    v = np.array([l1[0], l1[1]])-np.array([l2[0], l2[1]])
    distance = np.linalg.norm(v)
    return distance

def calculate_moving_average(landmark, ran, LiT):  
    while len(LiT) < ran:            
        LiT.append(landmark)
    LiT.append(landmark)             
    if len(LiT) > ran:               
        LiT.pop(0)
    return sum(LiT)/ran
def get_hand_label(index,hand,results):
    output =('Right 0.97', (1283, 827))
    for idx,classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            # process results 
            label=classification.classification[0].label
            score=classification.classification[0].score
            text='{} {}'.format(label,round(score,2))
            # extract coordinates
            coords=tuple(np.multiply(np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x,hand.landmark[mp_hands.HandLandmark.WRIST].y)),[1536,864]).astype(int))
            output=text,coords
    return output



def main(cap_device, kando):
    dis = 0.7                           
    preX, preY = 0, 0
    nowCli, preCli = 0, 0               
    norCli, prrCli = 0, 0               
    douCli = 0                          
    i, k, h = 0, 0, 0
    LiTx, LiTy, list0x, list0y, list1x, list1y, list4x, list4y, list6x, list6y, list8x, list8y, list12x, list12y = [
    ], [], [], [], [], [], [], [], [], [], [], [], [], []   
    nowUgo = 1
    cap_width = 1280
    cap_height = 720
    start, c_start = float('inf'), float('inf')
    c_text = 0
  
    video = cv.VideoCapture(cap_device)
    cfps = int(video.get(cv.CAP_PROP_FPS))
    if cfps < 30:
        video.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        video.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
        cfps = int(video.get(cv.CAP_PROP_FPS))
    ran = max(int(cfps/10), 1)
    hands = mp_hands.Hands(
        min_detection_confidence=0.8,   
        min_tracking_confidence=0.8,    
        max_num_hands=2
    )
    while video.isOpened():
        p_s = time.perf_counter()
        success, image = video.read()
        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        image.flags.writeable = False  
        results = hands.process(image)  

        image.flags.writeable = True    
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        landmarks_list = []
        if results.multi_hand_landmarks:
            hand_lands = results.multi_hand_landmarks[-1]
            for index, lm in enumerate(hand_lands.landmark):
                h, w, c = image.shape  # Height, Width, Channels
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append([index, cx, cy])

                # Drawing the Landmarks for only One Hand
                # Landmarks will be drawn for the Hand which was Detected First
                mp_drawing.draw_landmarks(image, hand_lands, mp_hands.HAND_CONNECTIONS)

            for num,hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),mp_drawing.DrawingSpec(color=(250,44,250),thickness=2,circle_radius=2),)
                get_hand_label(num,hand_landmarks,results)# to detetct left or right hand
                
                #render left right detection 
                if get_hand_label(num,hand_landmarks,results):
                    text,coords=get_hand_label(num,hand_landmarks,results)
                    cv.putText(image,text,coords,cv.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv.LINE_8)
                    if(text[0]=="R"):

                        landmark0 = [calculate_moving_average(hand_landmarks.landmark[0].x, ran, list0x), calculate_moving_average(
                            hand_landmarks.landmark[0].y, ran, list0y)]
                        landmark1 = [calculate_moving_average(hand_landmarks.landmark[1].x, ran, list1x), calculate_moving_average(
                            hand_landmarks.landmark[1].y, ran, list1y)]
                        landmark4 = [calculate_moving_average(hand_landmarks.landmark[4].x, ran, list4x), calculate_moving_average(
                            hand_landmarks.landmark[4].y, ran, list4y)]
                        landmark6 = [calculate_moving_average(hand_landmarks.landmark[6].x, ran, list6x), calculate_moving_average(
                            hand_landmarks.landmark[6].y, ran, list6y)]
                        landmark8 = [calculate_moving_average(hand_landmarks.landmark[8].x, ran, list8x), calculate_moving_average(
                            hand_landmarks.landmark[8].y, ran, list8y)]
                        landmark12 = [calculate_moving_average(hand_landmarks.landmark[12].x, ran, list12x), calculate_moving_average(
                            hand_landmarks.landmark[12].y, ran, list12y)]

                        absKij = calculate_distance(landmark0, landmark1)
                        absUgo = calculate_distance(landmark8, landmark12) / absKij
                        absCli = calculate_distance(landmark4, landmark6) / absKij

                        posx, posy = mouse.position

                        nowX = calculate_moving_average(
                            hand_landmarks.landmark[8].x, ran, LiTx)
                        nowY = calculate_moving_average(
                            hand_landmarks.landmark[8].y, ran, LiTy)

                        dx = kando * (nowX - preX) * image_width
                        dy = kando * (nowY - preY) * image_height
                        dx = dx+0.5
                        dy = dy+0.5
                        preX = nowX
                        preY = nowY
                        # print(dx, dy)
                        if posx+dx < 0:  
                            dx = -posx
                        elif posx+dx > screenRes[0]:
                            dx = screenRes[0]-posx
                        if posy+dy < 0:
                            dy = -posy
                        elif posy+dy > screenRes[1]:
                            dy = screenRes[1]-posy

              
                        if absCli < dis:
                            nowCli = 1          # nowCli: 1:click  0:non click
                            draw_circle(image, hand_landmarks.landmark[8].x * image_width,
                                        hand_landmarks.landmark[8].y * image_height, 20, (0, 250, 250))
                        elif absCli >= dis:
                            nowCli = 0
                        if np.abs(dx) > 7 and np.abs(dy) > 7:
                            k = 0                           
             
                        if nowCli == 1 and np.abs(dx) < 7 and np.abs(dy) < 7:
                            if k == 0:          
                                start = time.perf_counter()
                                k += 1
                            end = time.perf_counter()
                            if end-start > 1.5:
                                norCli = 1
                                draw_circle(image, hand_landmarks.landmark[8].x * image_width,
                                            hand_landmarks.landmark[8].y * image_height, 20, (0, 0, 250))
                        else:
                            norCli = 0
                        
                        # cursor
                        if absUgo >= dis and nowUgo == 1:
                            mouse.move(dx, dy)
                            
                            draw_circle(image, hand_landmarks.landmark[8].x * image_width,
                                        hand_landmarks.landmark[8].y * image_height, 8, (250, 0, 0))
                        # left click
                        if nowCli == 1 and nowCli != preCli:
                          
                            cv.putText(image, "Left click", (45, 365), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3) 
                            cv.putText(image, "For double left click do ", (33, 430),cv.FONT_HERSHEY_SIMPLEX, 1, (90, 233, 54), 3)
                            cv.putText(image, "left click twice in 0.5 sec", (33, 465),cv.FONT_HERSHEY_SIMPLEX, 1, (90, 233, 54), 3)

                            if h == 1:                                  
                                h = 0
                            elif h == 0:    
                                                      
                                mouse.press(Button.left)
                            # print('Click')
                        # left click release
                        if nowCli == 0 and nowCli != preCli:
                            mouse.release(Button.left)
                            k = 0
                            # print('Release')
                            if douCli == 0:                             
                                c_start = time.perf_counter()
                                douCli += 1
                            c_end = time.perf_counter()
                            if 10*(c_end-c_start) > 5 and douCli == 1: 
                                mouse.click(Button.left, 2)             # double click
                                douCli = 0
                        # right click
                        if norCli == 1 and norCli != prrCli:
                            # mouse.release(Button.left) 
                           
                            cv.putText(image, "Right Click", (45, 375), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)               
                            mouse.press(Button.right)
                            mouse.release(Button.right)
                            h = 1                                       
                            # print("right click")
                        # scroll
                        if hand_landmarks.landmark[8].y-hand_landmarks.landmark[5].y > -0.06:
                            mouse.scroll(0, -dy/50)
                           
                            cv.putText(image, "Scroll", (45, 375), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
                  
                            draw_circle(image, hand_landmarks.landmark[8].x * image_width,
                                        hand_landmarks.landmark[8].y * image_height, 20, (0, 0, 0))
                            nowUgo = 0
                        else:
                            nowUgo = 1

                        preCli = nowCli
                        prrCli = norCli
                    if(text[0]=="L"):
                            
                    # Stores 1 if finger is Open and 0 if finger is closed
                        fingers_open = []
                        if len(landmarks_list) != 0:
                            for tipId in fingerTipIds:
                                if tipId == 4:  # That is the thumb
                                    if landmarks_list[tipId][1] > landmarks_list[tipId - 1][1]:
                                        fingers_open.append(1)
                                    else:
                                        fingers_open.append(0)
                                else:
                                    if landmarks_list[tipId][2] < landmarks_list[tipId - 2][2]:
                                        fingers_open.append(1)
                                    else:
                                        fingers_open.append(0)

                        # Counts the Number of Fingers Open
                        count_fingers_open = fingers_open.count(1)

                        # If Hand Detected
                        flag=""
                        if results.multi_hand_landmarks != None:
       
                            if count_fingers_open == 1:
                                
                                cv.putText(image, "left", (45, 375), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
            

                                p.press('Left',presses=1)
            
            
                            if count_fingers_open ==5:
                               
                                cv.putText(image, "up", (45, 375), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                                p.press('Up',presses=1)
            

                            if count_fingers_open == 2:
                                
                                cv.putText(image, "right", (45, 375), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
            
                                p.press('Right',presses=1)
            

                            if count_fingers_open ==0:
                               
                                cv.putText(image, "down", (45, 375), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                                p.press('Down',presses=1)
                           
       
       
        p_e = time.perf_counter()
        fps = str(int(1/(float(p_e)-float(p_s))))
        cv.putText(image, "FPS:"+fps, (20, 80),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
     
        cv.imshow("Hand gesture",image)
        if (cv.waitKey(10) & 0xFF == ord('q')): 
            break
    video.release()



if __name__ == "__main__":
    
    main(cap_device, kando)

