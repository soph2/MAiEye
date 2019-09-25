import os
import argparse


#-------#
#Parsing#
#-------#
def main() :
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--labeltype", type=str, help="[string] : type of video. Is it map of mob?")
    ap.add_argument("-v", "--video", type=str, help="[string] : path to input video file")
    ap.add_argument("-t", "--tracker", type=str, help="[string] : BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT")
    ap.add_argument("-l", "--label", type=str, help="[string] : label name")
    ap.add_argument("-ms", "--milliseconds", type=int, help="[int] : save image per n milliseconds")
    ap.add_argument("-sp", "--savepath", type=str, help="[string] : saving path (absolute path , 절대경로), 경로가 존재하지 않으면 저장되지 않을 수 있어요!")
    ap.add_argument("-is", "--imagesize", type=int, help="[int] : map 옵션 을 선택할 때에만 지정. N * N 으로 사진이 저장됨")
    args = vars(ap.parse_args())

    videoPath = args["video"]
    trackerType = args["tracker"]
    labelName = args["label"]
    savefolder = args["savepath"]
    waitingTime = args["milliseconds"]

    # 맵 캡쳐 모듈 호출
    if args['labeltype'] == 'map' :
        size = args["imagesize"]
        from JanghooModule_MapCapture.capture_map import captureMapByFrame
        if os.path.isdir(videoPath):
            file_list = os.listdir(videoPath)
            for filename in file_list:
                newvideopath = os.path.join(videoPath, filename)
                captureMapByFrame(size, newvideopath, labelName, savefolder, waitingTime)
        else:
            captureMapByFrame(size, videoPath, labelName, savefolder, waitingTime)

    # 관심영역 추출 모듈 호출
    if args['labeltype'] == 'mob' :
        from JanghooModule_RoIExtraction.roi_multi_tracking_in_video import flow
        print("\n\nmob session start\n\n")
        flow(videoPath, trackerType, labelName, savefolder, waitingTime)
        print("\n\nmob session end\n\n")


# 모듀
if __name__ == '__main__' :
    print("\n\n\nmain in\n\n\n")
    main()
    print("\n\n\nmain end\n\n\n")