# coding:utf-8
import RPi.GPIO as GPIO
import numpy as np
import cv2
from matplotlib import pyplot as plt

status = 0

key_run = 12
key_set = 16
key_display = 18

def key_runInterrupt(key_run):
    print('key run press')
    status = 1

def key_setInterrupt(key_set):
    print('key set press')
    status = 2

def key_displayInterrupt(key_display):
    print('key display press')
    status =3

GPIO.setmode(GPIO.BOARD)
GPIO.setup(key_run,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(key_set,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(key_display,GPIO.IN,pull_up_down=GPIO.PUD_UP)

GPIO.add_event_detect(key_run,GPIO.FALLING,key_runInterrupt,400)
GPIO.add_event_detect(key_set,GPIO.FALLING,key_setInterrupt,400)
GPIO.add_event_detect(key_display,GPIO.FALLING,key_displayInterrupt,400)

MIN_MATCH_COUNT = 10
camera = cv2.VideoCapture(0)
update_img = 0
img2 = cv2.imread('template.jpg', 1)  # queryImage
while True:
    print(status)
    key = cv2.waitKey(1) & 0xFF
    (grabbed, frame) = camera.read()
    cv2.imshow('frame', frame)
    # Wait key press down
    # 如果q键被按下，跳出循环
    if key == ord("q"):
        break

    #if key == ord("s"):  # 设置一个标签，当有运动的时候为1
    if status == 2:
        print("Update img")
        cv2.imwrite("template.jpg", frame)
        update_img = 1
        status = 0

    #if key == ord("c"):  # 设置一个标签，当有运动的时候为1
    if status == 1:
        status = 0
        img1 = frame
        if update_img is 1:
            img2 = cv2.imread('template.jpg', 1)  # queryImage
            print("Update img successfully")
        # img1 = cv2.imread('template.jpg',0) # trainImage

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w ,d = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            print "Matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        else:
            print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
            matchesMask = None

        # Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed).

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        # plt.imshow(img3, 'gray'), plt.show()
camera.release()
cv2.destroyAllWindows()
