import cv2
import sys
import os
import argparse


# 파일에 대한 속성을 저장하고, 그 속성을 바탕으로 파일을 작성하는 기능을 가진 클래스
class FileData :
    def __init__(self, filefullpath, video_frame_count, frame, savefolder, labelname):
        self.file_fullpath = filefullpath
        self.file_folderpath,   self.file_fullname = os.path.split(filefullpath)
        self.file_justfilename, self.file_justext  = os.path.splitext(self.file_fullname)
        self.file_savefolder = savefolder
        self.video_frame_count = video_frame_count
        self.labelname = labelname

        self.xml_file_name = str(self.file_justfilename) + "_" + str(labelname) + "_" + str(self.video_frame_count) + '.xml'
        self.img_file_name = str(self.file_justfilename) + "_" + str(labelname) + "_" + str(self.video_frame_count) + '.jpg'

    def writeAndSave(self, image):
        cv2.imwrite(os.path.join(self.file_savefolder, self.img_file_name), image)


    # save folder 내부에, label 이라는 이름의 하위 폴더를 추가하여 해당 파일 안에 저장한다.
    def writeAndSaveWithLabel(self, image):
        labelpath = os.path.join(self.file_savefolder, self.labelname)
        if not os.path.isdir(labelpath) :
            os.makedirs(labelpath)
        cv2.imwrite(os.path.join(labelpath, self.img_file_name), image)



# 실제 프로그램 흐름을 가진 모듈
def captureMapByFrame(size, videoPath, labelName, savefolder, waitingTime) :

    cap = cv2.VideoCapture(videoPath)
    success, frame = cap.read()


    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)


    if not os.path.isdir(savefolder) :
        # save folder 이 존재하지 않는다면..
        os.makedirs(savefolder)
        print("(-sp --savepath ::) 폴더가 존재하지 않아, 새로운 폴더를 생성합니다.")
        # 폴더를 생성한다.


    save_image_per_n_milliseconds = waitingTime
    framecount = 0
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break
        framecount += 1

        frame = cv2.resize(frame, dsize=(size, size), interpolation=cv2.INTER_AREA)
        filedata = FileData(videoPath, framecount, frame, savefolder, labelName)


        # show frame
        cv2.imshow('ImageSaver', frame)


        # -ms 로 지정한 만큼마다, 그 프레임의 데이터와 사진을 저장함.
        if framecount % save_image_per_n_milliseconds == 0 :
            filedata.writeAndSaveWithLabel(frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

    print("session end")


if __name__ == '__main__' :
    print('called module separately')

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--labeltype", type=str, help="[string] : type of video. Is it map of mob?")
    ap.add_argument("-v", "--video", type=str, help="[string] : path to input video file")
    ap.add_argument("-l", "--label", type=str, help="[string] : label name")
    ap.add_argument("-ms", "--milliseconds", type=int, help="[int] : save image per n milliseconds")
    ap.add_argument("-sp", "--savepath", type=str, help="[string] : saving path (absolute path , 절대경로), 경로가 존재하지 않으면 저장되지 않을 수 있어요!")
    ap.add_argument("-is", "--imagesize", type=int, help="[int] : map 옵션 을 선택할 때에만 지정. N * N 으로 사진이 저장됨")
    args = vars(ap.parse_args())


    videoPath = args["video"]
    labelName = args["label"]
    savefolder = args["savepath"]
    waitingTime = args["milliseconds"]
    size = args["imagesize"]


    if os.path.isdir(videoPath):
        file_list = os.listdir(videoPath)
        for filename in file_list:
            newvideopath = os.path.join(videoPath, filename)
            captureMapByFrame(size, newvideopath, labelName, savefolder, waitingTime)
    else:
        captureMapByFrame(size, videoPath, labelName, savefolder, waitingTime)