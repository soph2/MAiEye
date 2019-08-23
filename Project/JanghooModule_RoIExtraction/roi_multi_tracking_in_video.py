import sys
import cv2

from random import randint
import argparse


#-------#
#Parsing#
#-------#
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, help="BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT")

args = vars(ap.parse_args())


# Set video to load
videoPath = args["video"]


#reference https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
trackerTypes =['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()


# quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)

#namedWindow : string 이름을 가진 window 를 표시한다.
#imshow : 특정 window 에 img 를 표시한다.
cv2.namedWindow('Selecting RoI')
cv2.imshow('Selecting RoI', frame)

## Select boxes
bboxes = []
colors = []


# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
count = 0
while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    print("\n\n현재 저장된 boundary box 개수 : {}".format(count))
    bbox = cv2.selectROI('Selecting RoI', frame)
    print("\n\n")


    print("object 선택을 마치고 tracking 을 시작하려면 q를 누르세요.")
    print("다음 object 를 선택하려면 아무 키나 누르세요.")
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    count += 1
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
        break

print('Selected bounding boxes {}'.format(bboxes))

# Specify the tracker type
trackerType = args["tracker"]

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)



#----------------------------------------#
#Show original boundingbox with Subwindow#
#----------------------------------------#
frameandBbox = frame
for bbox, color in zip(bboxes, colors):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frameandBbox, pt1 = p1, pt2 = p2, color = color, thickness= 2)
    print(p1, p2, color)
cv2.namedWindow('Selecting RoI')
cv2.imshow('Selecting RoI', frameandBbox)




#--------------#
#Save the Movie#
#--------------#
w = cap.get(3)
h = cap.get(4)
output_size = (int(w), int(h))
print(output_size)
codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (videoPath.split('.')[0]), codec, cap.get(cv2.CAP_PROP_FPS), output_size)
#cap.get(cv2.CAP_PROP_FPS) : get - cap 에서 가져와. prop_fps : 적당한 frame per second를. 즉, 그 동영상의 fps
#output_size : 위에서 설정한 hyper parameter 임.



# Process video and track objects
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # 그냥 박스 안에 무슨 요소들이 들어가 있는지 궁금해서...
    print(boxes)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # show frame
    cv2.imshow('MultiTracker', frame)
    out.write(frame)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break

