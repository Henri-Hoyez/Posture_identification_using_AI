from enum import Enum

class Pose(Enum):
    laying = 0
    sitting = 1
    standing = 2
    unknown = 3

class Action(Enum):
    clapping = 0
    dabbing = 1
    fall_floor = 2
    idle = 3
    slaping = 4
    walking = 5
    unknown = 6

class Gait(Enum):
    Baptiste = 0
    Henri = 1
    Jerome = 2
    Lucas = 3
    Nicolas = 4
    Unknown = 5


def getPoseFromId(_id):
    return Pose(_id).name

def getActionFromId(_id):
    return Action(_id).name

def getGaitFromId(_id):
    return Gait(_id).name

def getPoseFromName(name):
    return Pose(name).value

def getActionFromName(name):
    return Action(name).value

def getGaitFromName(name):
    return Gait(name).value