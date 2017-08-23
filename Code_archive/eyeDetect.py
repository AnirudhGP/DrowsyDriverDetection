import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('C:\Program Files\Anaconda3\pkgs\opencv3-3.1.0-py35_0\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\Program Files\Anaconda3\pkgs\opencv3-3.1.0-py35_0\Library\etc\haarcascades\haarcascade_eye.xml')

for i in range(0, 1520):
    print(i)
    pos =""
    if int(i/10) == 0:
        pos = "000" + str(i)
    elif int(i/100) == 0:
        pos = "00" + str(i)
    elif int(i/1000) == 0:
        pos = "0" + str(i)
    else:
        pos = str(i)

    img = cv2.imread('Dataset/BioID_' + pos + '.pgm')
    #img = cv2.imread('me.jpg')
    if img.shape[0] < 500 or img.shape[1] < 500:
        img = cv2.resize(img, (0, 0), fx=2, fy=2)
    #img = cv2.resize(img, (800, 800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('FacesGray/BioID_' + pos + ".jpg", gray)
    '''faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        eyePos = []
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eyeNo = 0
        for (ex,ey,ew,eh) in eyes:
            #print(eh, ew)
            if ew < 50 or eh < 50:
                continue
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            im = roi_gray[ey:ey + eh, ex:ex + ew]
            im = cv2.resize(im, (100, 100))
            #cv2.imshow('brow', im)
            cv2.imwrite('Eyes/eye' + pos + "_" + str(eyeNo) + ".jpg", im)
            eyeNo += 1
        print("Completed " + str(i) + " out of 1520")
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        ret, im = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)
        im = 255 - im
        cv2.imshow('thresh', im)
        kernel = np.array([[0, 0, 1, 0, 0],
                           [0, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 0, 0]]).astype('uint8')
        im = cv2.dilate(im, kernel, iterations=1)

        cv2.imshow('dilate', im)
        #cv2.imshow('before', im)
        #im = cv2.Canny(im, 100, 200)
        im = 255 - im
        cv2.imshow('eye', im)
        cv2.waitKey(0)
        circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 2, 10, param1=30, param2=30, minRadius=0, maxRadius=20)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(im, (i[0], i[1]), i[2], (127, 127, 127), 2)
                # draw the center of the circle
                cv2.circle(im, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imshow('detected circles', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()