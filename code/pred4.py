import cv2
import numpy as np
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import pyttsx3

visual_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b',
               12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm',
               23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r',
               28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'}
output = load_model('ISLgood1.h5')
counter = 0
indexlist = []

'''Finding Average'''
bg = None

num_frames = 0
aWeight = 0.5

global finalstring
finalstring = ""


def show():
    return finalstring


def run_avg(image, aweight):
    global bg
    if bg is not None:
        cv2.accumulateWeighted(image, bg, aweight)
        return
    bg = image.copy().astype("float")


def extract_hand(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        max_cont = max(cnts, key=cv2.contourArea)
        return thresh, max_cont


def fun(frame, clear_text=False):
    global num_frames, indexlist, finalstring

    if clear_text:
        finalstring = ""  # Reset the text

    cv2.rectangle(frame, (300, 150), (30, 400), (255, 255, 255), 3)
    roi = frame[150:400, 30:300]
    detector = HandDetector(maxHands=1)
    hands = detector.findHands(roi, draw=False)
    # roi = cv2.flip(roi, 1)
    gb = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gb = cv2.GaussianBlur(gb, (7, 7), 0)

    if num_frames < 10:
        run_avg(gb, aWeight)
        cv2.putText(frame, "Keep the Camera still.", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0))

    else:
        cv2.putText(frame, "Press esc to exit.", (10, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        hand = extract_hand(gb)

        if hand is not None:
            thresh, max_cont = hand
            mask = cv2.drawContours(frame, [max_cont + (30, 150)], -1, (0, 0, 255))
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [max_cont], -1, 255, -1)
            mask = cv2.medianBlur(mask, 5)
            mask = cv2.addWeighted(mask, 0.5, mask, 0.5, 0.0)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            res = cv2.bitwise_and(roi, roi, mask=mask)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            high_thresh, thresh_im = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            lowthresh = 0.5 * high_thresh
            res = cv2.Canny(res, lowthresh, high_thresh)
            a = cv2.resize(res, (100, 100))
            a = np.array(a)
            a = a.reshape((-1, 100, 100, 1))
            a.astype('float32')
            a = a / 255.0
            if hands:
                pred = output.predict(a)
                sign = np.argmax(pred)
                final_sign = visual_dict[sign]
                indexlist.append(str(final_sign))
                cv2.putText(frame, 'Sign ' + str(final_sign), (10, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))

                sublist = []
                for i in range(1, len(indexlist)):
                    if indexlist[i] == indexlist[i - 1]:
                        sublist.append(indexlist[i])
                        if len(sublist) == 5:
                            finalstring += indexlist[i]
                            sublist = []
                            indexlist = []
                        else:
                            pass
                    else:
                        sublist = []
    num_frames += 1
    return frame, finalstring


def ind():
    pyobj = pyttsx3.init()
    pyobj.say(str(finalstring))
    pyobj.setProperty("rate", 170)
    pyobj.setProperty("volume", 1)
    pyobj.runAndWait()
