import cv2
import numpy

import argparse



#-------#
#Parsing#
#-------#
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
args = vars(ap.parse_args())



#video_path = "yongjae.mp4"
video_path = args["video"]
output_size = (0,0) #(width, height)
cap = cv2.VideoCapture(video_path)

# VideoCapture
# The function reads an image from the specified buffer in the memory.
# If the buffer is too short or contains invalid data, the empty matrix/image is returned.

if not cap.isOpened():
    exit()

tracker = cv2.TrackerCSRT_create()


#첫 번째 프레임을 읽어온다.
ret, img = cap.read()

#namedWindow : string 이름을 가진 window 를 표시한다.
#imshow : 특정 window 에 img 를 표시한다.
cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)


#-----------#
#Setting ROI#
#-----------#


# In the C++ version, selectROI allows you to obtain multiple bounding boxes,
# but in the Python version, it returns just one bounding box.
# So, in the Python version, we need a loop to obtain multiple bounding boxes.
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')


#------------------#
#initialize tracker#
#------------------#
tracker.init(img, rect)





#--------------#
#Save the Movie#
#--------------#
w = cap.get(3)
h = cap.get(4)
print(w, h)
output_size = (int(w), int(h))
print(output_size)
codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), codec, cap.get(cv2.CAP_PROP_FPS), output_size)
#cap.get(cv2.CAP_PROP_FPS) : get - cap 에서 가져와. prop_fps : 적당한 frame per second를. 즉, 그 동영상의 fps
#output_size : 위에서 설정한 hyper parameter 임.




#--------------#
#Frame by Frame#
#--------------#
while True :
    ret, img = cap.read()

    if not ret :
        exit()

    success, box = tracker.update(img)
    left, top, w, h = [int(v) for v in box]
    cv2.rectangle(img, pt1=(left,top), pt2=(left + w, top + h), color=(255, 255, 255), thickness=3)

    cv2.imshow('img', img)

    out.write(img)

    # Key 입력을 1 msec만큼 기다린다.
    if cv2.waitKey(1) == ord('q') :
        break

out.release()


