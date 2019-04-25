from enum import Enum



class Gait(Enum):
    Baptiste = 0
    Henri = 1
    Nicolas = 2



def getGaitFromId(_id):
    return Gait(_id).name

def getGaitFromName(name):
    return Gait(name).value