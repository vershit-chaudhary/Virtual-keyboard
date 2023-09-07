import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 1800)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O","P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finaltext=""
#def drawAll(img, buttonList):
#    for button in buttonList:
#        cv2.rectangle(img, button.pos, (button.pos[0] + 46, button.pos[1] + 50), (255, 0, 255), cv2.FILLED)
#        cv2.putText(img, button.text, (button.pos[0] + 1, button.pos[1] + 48), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255),
#                3)
#    return img##

def drawAll(img, buttonList):
    imgNew=np.zeros_like(img,np.uint8)

    for button in buttonList:
        cvzone.cornerRect(imgNew,(button.pos[0], button.pos[1], 46, 50),10, 2,rt=0)
        cv2.rectangle(imgNew, button.pos, (button.pos[0] + 46, button.pos[1] + 50), (240, 0, 0), cv2.FILLED)
        cv2.putText(imgNew, button.text, (button.pos[0] + 1, button.pos[1] + 48), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255),
                3)
    out=img.copy()
    alpha=0.2
    mask=imgNew.astype(bool)
    #print(mask.shape)
    out[mask]=cv2.addWeighted(img,alpha,imgNew,1-alpha,0)[mask]
    return out
class Button():
    def __init__(self, pos, text):
        self.pos = pos
        self.text = text


buttonList = []
for j in range(0, 3):
    for i, key in enumerate(keys[j]):
        buttonList.append(Button([20 + i * 60, 100 + j * 60], key))
while True:
    success, img = cap.read()
    img=cv2.flip(img,1)
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)
    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            if button.pos[0]<lmList[8][0]<button.pos[0]+46 and button.pos[1]<lmList[8][1]<button.pos[1]+50:
                cv2.rectangle(img, button.pos, (button.pos[0] + 46, button.pos[1] + 50), (240, 0, 0), cv2.FILLED)
                cv2.putText(img, button.text, (button.pos[0] + 1, button.pos[1] + 48), cv2.FONT_HERSHEY_PLAIN, 4,
                            (255, 255, 255),3)
                l,_,_=detector.findDistance(8, 12,img)
                if l<=30:
                    cv2.rectangle(img, button.pos, (button.pos[0] + 46, button.pos[1] + 50), (250, 0, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (button.pos[0] + 1, button.pos[1] + 48), cv2.FONT_HERSHEY_PLAIN, 4,
                                (0, 0, 0), 3)
                    finaltext+=button.text
                    sleep(0.20)

    cv2.rectangle(img, (20, 300), (700, 500), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, finaltext, (25,325), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),
                1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


