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
        for ip in args.ip_camera:
            IP_CAMERAS.append( ip )
        
        print(IP_CAMERAS)
          
    

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

def openAndRecord(camNumber, event, record_event):
    print(camNumber)

    isRecording = False

    cap = cv2.VideoCapture(camNumber)


    int(cap.set(cv2.CAP_PROP_FPS,30))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = None   
    out = None 

    print("[+] video size: ", width,"*", height, "at ",fps,"FPS")
    ret = True

    while(ret and not event.is_set()):
        ret, frame = cap.read()
        cv2.imshow('Camera_'+str(camNumber), frame)

        if record_event.is_set():
            if(isRecording):
                out.write(frame)
            else:
                isRecording = True
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                video_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_CAMERA_"+ str(camNumber)
                out = cv2.VideoWriter(VIDEO_OUT_PATH+"\\"+video_name.replace("/","_").replace(".","_").replace(":","_")+ ".mp4",fourcc, fps, (width,height))
                print(VIDEO_OUT_PATH+"\\"+video_name.replace("/","_").replace(".","_").replace(":","_")+ ".mp4")
                out.write(frame)
                print("[+] Camera ", camNumber, "start recording.")
        else:
            if isRecording:
                isRecording = False
                fourcc = None
                out = None
                print("[+] Camera ", camNumber, "stop recording.")


        c = cv2.waitKey(1)

        if c & 0xFF == ord('q'):
            event.set()
            break
        elif c & 0xFF == ord('r'):
            if(record_event.is_set()):
                record_event.clear()
            else:
                record_event.set()

    if(out is not None):
        out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    getArguments()

    cam_number = DeviceCounter()

    event = Event()
    record_event = Event()

    print(cam_number)
    for i in range(cam_number+1):
        p = Process(target=openAndRecord, args=(i,event,record_event,))
        p.start()
    if(len(IP_CAMERAS) != 0 ):
        for c in IP_CAMERAS:
            print(c)
            p = Process(target=openAndRecord, args=(c,event,record_event,))
            p.start()



