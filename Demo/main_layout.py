import cv2 

def setMainLayout(windows_properties, frame, title, fps, display_help, isInDevMode):
    frame = setTitle(windows_properties, frame, title)
    frame = setFPS(windows_properties, frame, fps)
    if display_help:
        frame = setHelp(windows_properties, frame, isInDevMode)
    return frame

def setTitle(windows_properties, frame, title):
    #Display title
    x,y,w,h = windows_properties
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    (text_width, text_height) = cv2.getTextSize(title, font, fontScale=font_scale, thickness=1)[0]
    
    img = frame.copy()
    img = cv2.rectangle(img, (int(w/2-text_width/2-10),text_height+45), (int(w/2+text_width/2+5),20), (255,255,255), cv2.FILLED)
    frame = cv2.addWeighted(img, .3, frame, .7, 0)
    
    frame = cv2.putText(frame,title,(int(w/2-text_width/2),text_height+30), font, font_scale,(20,30,0),2,cv2.LINE_AA)
    
    return frame

def setFPS(windows_properties, frame, fps):
    #Display FPS
    x,y,w,h = windows_properties
    
    text_fps = 'FPS : '+str(int(fps))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .8
    (text_width, text_height) = cv2.getTextSize(text_fps, font, fontScale=font_scale, thickness=1)[0]
    
    img = frame.copy()
    img = cv2.rectangle(img, (w-text_width-30,h-10), (w-10,h-text_height-30), (255,255,255), cv2.FILLED)
    frame = cv2.addWeighted(img, .5, frame, .5, 0)
    
    frame = cv2.putText(frame,text_fps,(w-text_width-20,h-20), font, font_scale,(20,30,0),2,cv2.LINE_AA)
    
    return frame

def setHelp(windows_properties, frame, isInDevMode):
    #Display help
    x,y,w,h = windows_properties

    text = "'h' : Display/Hide Help"
    if isInDevMode:
        text = text + "\n'd' : Begin saving standing data\n'a' : Begin saving sitting data\n'c' : Begin saving laying data\n's' : Stop saving data"
    text = text + "\n'q' : Quit application"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .3

    nb_lines = len(text.split("\n"))
    max_width = 0
    text_height = 0
    sum_heights = 0
    lines = text.split("\n")
    datas = []
    for i in range(nb_lines):
        (text_width, text_height) = cv2.getTextSize(lines[i], font, fontScale=font_scale, thickness=1)[0]
        sum_heights += text_height + 10
        max_width = text_width if text_width > max_width else max_width
        datas.append([text_width,text_height])
    sum_heights -= 10

    img = frame.copy()
    img = cv2.rectangle(img, (5,int(h/2+sum_heights/2)+10), (max_width+30,int(h/2-sum_heights/2-5)), (255,255,255), cv2.FILLED)
    frame = cv2.addWeighted(img, .5, frame, .5, 0)

    for i in range(len(lines)):
        frame = cv2.putText(frame,lines[i],(15,int(h/2-sum_heights/2+(i+1)*(text_height+5))), font, font_scale,(20,30,0),1,cv2.LINE_AA)
    
    return frame