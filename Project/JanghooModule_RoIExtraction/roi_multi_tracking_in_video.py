import sys
import cv2
import os

from random import randint
import argparse
from BBoxToDataset.CreateTracker import createTrackerByName


#-------#
#Parsing#
#-------#
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="[string] : path to input video file")
ap.add_argument("-t", "--tracker", type=str, help="[string] : BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT")
ap.add_argument("-l", "--label", type=str, help="[string] : label name")
ap.add_argument("-ms", "--milliseconds", type=int, help="[int] : save image per n milliseconds")
ap.add_argument("-sp", "--savepath", type=str, help="[string] : saving path (absolute path , 절대경로)")
args = vars(ap.parse_args())
# Set video to load
videoPath = args["video"]
# Specify the tracker type
trackerType = args["tracker"]
# and the others...
labelName = args["label"]
savefolder = args["savepath"]
waitingTime = args["milliseconds"]



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

print('Selected bounding boxes : {}'.format(bboxes))



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




#---------------------#
#Movie Saving Settings#
#---------------------#
w = cap.get(3)
h = cap.get(4)
output_size = (int(w), int(h))
print(output_size)
codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
outconfig = cv2.VideoWriter('%s_output.mp4' % (videoPath.split('.')[0]), codec, cap.get(cv2.CAP_PROP_FPS), output_size)
#cap.get(cv2.CAP_PROP_FPS) : get - cap 에서 가져와. prop_fps : 적당한 frame per second를. 즉, 그 동영상의 fps
#output_size : 위에서 설정한 hyper parameter 임.




#-----------------------------------------#
#Imagefile and Annotatioin Saving Settings#
#-----------------------------------------#

from BBoxToDataset.BboxObject import BboxObject
    # bbox object 객체들을 보관할것
    # bbox 가 4개라면 bbox 객체 4개를 만드는것
bboxobjects = []
for i in range(0,len(bboxes),1) :
    a = BboxObject((0,0), (0,0), labelName)
    bboxobjects.append(a)

from BBoxToDataset.BboxObject import FileData
if not os.path.isdir(savefolder) :
    # save folder 이 존재하지 않는다면..
    os.makedirs(savefolder)
    print("폴더가 존재하지 않아, 새로운 폴더를 생성합니다.")
    # 폴더를 생성한다.

#class init parameters
filefullpath = videoPath
savefolder = savefolder


#-------------------------------#
#Process video and track objects#
#-------------------------------#
font = cv2.FONT_HERSHEY_SIMPLEX #just font setting
save_image_per_n_milliseconds = waitingTime
framecount = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    framecount += 1
    originalframe = frame.copy()
    filedata = FileData(filefullpath, framecount, frame, savefolder)

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # 그냥 박스 안에 무슨 요소들이 들어가 있는지 궁금해서...
    # print(boxes)


    # draw tracked objects
    # 만약 object 가 4개라면 loop 를 4번 돌게 된다.
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        # Set Font and put Text on BBox
        cv2.putText(frame, labelName,
                    (int(newbox[0]), int(newbox[1] - 4)),
                    font,
                    fontScale=0.5,
                    color=colors[i],
                    thickness=2,
                    lineType=cv2.LINE_AA)

        if framecount % save_image_per_n_milliseconds == 0 :
            # -ms 로 지정한 만큼마다, bbox object 객체의 데이터를 초기화해줌.
            # 초기화된 데이터는 object 객체에 들어가줘야함.
            bboxobjects[i].__init__(p1, p2, labelName)
            filedata.setObject(bboxobjects[i])

    # show frame
    cv2.imshow('MultiTracker', frame)
    outconfig.write(frame)

    # -ms 로 지정한 만큼마다, 그 프레임의 데이터와 사진을 저장함.
    if framecount % save_image_per_n_milliseconds == 0 :
        filedata.writeAndSave(originalframe)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break

print("session end")



#reference https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/

