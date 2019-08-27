import cv2
import sys
import os


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
        # print(os.path.join(self.file_savefolder, self.img_file_name))
        cv2.imwrite(os.path.join(self.file_savefolder, self.img_file_name), image)


def captureMapByFrame(size, videoPath, labelName, savefolder, waitingTime) :
    cap = cv2.VideoCapture(videoPath)
    success, frame = cap.read()


    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)



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
            filedata.writeAndSave(frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

    print("session end")
