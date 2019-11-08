import cv2
VIDEONAME = "output.avi"
cap = cv2.VideoCapture(VIDEONAME)
hf = True
cnt = 0
while hf:
    hf, f = cap.read()
    cv2.imwrite("output/%s.jpg"%cnt, f)
    cnt += 1
