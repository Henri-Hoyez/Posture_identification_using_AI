import argparse

def GetParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_mode", action='store_true', help="Enter in developer mode")
    parser.add_argument("-d","--df", default="./datasets", type = str, help="Asks for the path folder where the user wants to store the dataset (Only available for pose recognition)")
    parser.add_argument("-t","--thermic", action='store_true', help="Tells the program that we use a thermical camera")
    parser.add_argument("-ns","--no-skeleton", action='store_true', help="Tells the program that we don't want to show the skeleton")
    parser.add_argument("-o","--out", type =str, help="Sets the directory to store the video recorded")
    parser.add_argument("--mode", default="webcam", type=str, help="Mode selection (webcam, pose, action, gait, skeleton)")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../../models/"

    # Add others in path
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    args = args[0].__dict__
    
    return args, params