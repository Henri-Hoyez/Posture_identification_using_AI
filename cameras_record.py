from multiprocessing import Process, Lock, Event
import argparse
import datetime
import cv2

VIDEO_OUT_PATH = "C:\\Users\\Henri HO\\Desktop\\tes lstm\\videos"
IP_CAMERAS = []

    
def getArguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ip_camera',nargs='*',  help='<Optional> Set camera ip Address')
    args = parser.parse_args()
    print(args)
    if args.ip_camera is not None:
        IP_CAMERAS = args.ip_camera    
    

def DeviceCounter():
    i=1
    a=True
    ctr=0
    while(a):
        cap=cv2.VideoCapture(i)
        ret=cap.isOpened()
        if(ret):
            ctr+=1
            i+=1
            a=True
        else:
            a=False
    return ctr

def openAndRecord(camNumber, event):

    video_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_CAMERA_"+ str(camNumber)+ ".mp4"

    cap = cv2.VideoCapture(camNumber)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(VIDEO_OUT_PATH+"\\"+video_name,fourcc, 30, (640,480))

    print("[+] video size: ", height,"*", width)
    ret = True

    while(ret and not event.is_set()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            event.set()
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    getArguments()

    cam_number = DeviceCounter()
    event = Event()

    for i in range(cam_number):
        p = Process(target=openAndRecord, args=(i+1,event,))
        p.start()
    if(len(IP_CAMERAS) is not 0):
        for c in IP_CAMERAS:
            p = Process(target=openAndRecord, args=(c,event,))
            p.start()



