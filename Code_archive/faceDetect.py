import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('C:\Program Files\Anaconda3\pkgs\opencv3-3.1.0-py35_0\Library\etc\haarcascades\haarcascade_frontalface_alt2_16layers.xml')
eye_cascade = cv2.CascadeClassifier('C:\Program Files\Anaconda3\pkgs\opencv3-3.1.0-py35_0\Library\etc\haarcascades\haarcascade_eye.xml')

dirs = ['Dataset/yawnFace', 'Dataset/normalFace']

def detectFaces():
    pos = 0
    for dir in dirs:
        cnt = 0
        facecnt = 0
        for filename in os.listdir(dir):
            if filename.endswith('.png'):
                im = cv2.imread(dir + '/' + filename)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #gray = gray[:, 120:gray.shape[1]-80]
                #gray = cv2.equalizeHist(gray)
                #gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                # cv2.imshow('edge', edges)
                # cv2.imshow('s', gray)
                flag = 0
                # cv2.imshow('x', im)
                # cv2.waitKey(0)
                faces = face_cascade.detectMultiScale(gray, 1.01, 5, minSize=(60, 60))
                cnt += 1
                for (x, y, w, h) in faces:
                    if w < 150 or h < 150:
                        continue
                    flag = 1
                    facecnt += 1
                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    eyePos = []
                    roi_gray = gray[y-10:y + h + 10, x - 10:x + w + 10]
                    roi_color = im[y:y + h, x:x + w]
                    cv2.imwrite('Dataset/detectedFaces/IMG_' + str(pos) + '.png', roi_gray)
                    pos += 1
                    #print(roi_gray.shape)
                    #cv2.waitKey(0)
               # if flag == 0:
                    # cv2.imshow('asd', gray)
                    # cv2.waitKey(0)
               # else:
                 #   cv2.imwrite('sampleFaceWorking.png', gray)
        print(cnt, facecnt)

detectFaces()
