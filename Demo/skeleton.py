import cv2
import numpy as np 


class Skeleton():
    
    instance = None

    joints = [
        (0,1),
        (1,8),
        
        (1,2),
        (2,3),
        (3,4),
        
        (1,5),
        (5,6),
        (6,7),

        (0,15),
        (15,17),
        (0,16),
        (16,18),

        (8,9),
        (9,10),
        (10,11),

        (8,12),
        (12,13),
        (13,14),

        (11,24),
        (11,22),
        (22,23),

        (14,21),
        (14,19),
        (19,20)
    ]

    def __new__(self, **kwargs):
        if not Skeleton.instance:
            Skeleton.instance = Skeleton.__Skeleton(**kwargs)
        return Skeleton.instance

    class __Skeleton():

        def __init__(self, **kwargs):
            self.setValues(**kwargs)

        def setValues(self, color_joints=(0,150,50), color_points=(0,0,0), radius=8, factor=0.65):
            self.color_joints = color_joints
            self.color_points = color_points
            self.points_radius = radius
            self.factor = factor

        # keypoints: [25x3]
        def cropSkeleton(self, keypoints, width, height):
            cropped = np.array(keypoints)
            min_x = cropped[0,0]
            min_y = cropped[0,1]
            min_z = cropped[0,2]
            idxs_not_found = []
            for i in range(cropped.shape[0]):
                x, y, z = cropped[i]
                if x == 0 and y == 0 and z == 0:
                    idxs_not_found.append(i)
                    continue
                min_x = x if x < min_x else min_x
                min_y = y if y < min_y else min_y
                min_z = z if z < min_z else min_z
            cropped[:,0] -= min_x
            cropped[:,1] -= min_y
            cropped[:,2] -= min_z
            max_dist_x = np.max(cropped[:,0])
            max_dist_y = np.max(cropped[:,1])

            if max_dist_y > max_dist_x:
                cropped /= max_dist_y
                cropped *= height * self.factor
                cropped[:,1] += 0.5 * (1-self.factor) * height
                width_cropped = np.max(cropped[:,0])
                cropped[:,0] += 0.5 * (width - width_cropped)
            else:
                cropped /= max_dist_x
                cropped *= width * self.factor
                cropped[:,0] += 0.5 * (1-self.factor) * width
                height_cropped = np.max(cropped[:,1])
                cropped[:,1] += 0.5 * (height - height_cropped)

            return cropped, idxs_not_found


        def drawSkeletonCropped(self, keypoints, window_size, frame):
            x,y,w,h = window_size
            skeleton, idxs_not_found = self.cropSkeleton(keypoints, w, h)
            for joint in Skeleton.joints:
                pt1, pt2 = joint
                if pt1 in idxs_not_found or pt2 in idxs_not_found :
                    continue
                cv2.line(frame, (skeleton[pt1,0], skeleton[pt1,1]), (skeleton[pt2,0], skeleton[pt2,1]), self.color_joints, 5)
                #cv2.line(image, (x1, y1), (x2, y2), (0,255,0), lineThickness)
            for i in range(skeleton.shape[0]):
                x, y, z = skeleton[i]
                if i in idxs_not_found:
                    continue
                cv2.circle(frame, (x, y), self.points_radius, self.color_points, -1)
            return frame


        def drawSkeleton(self, keypoints, window_size, frame): # (?,25,3)
            x,y,w,h = window_size
            for person in keypoints:
                for joint in Skeleton.joints:
                    pt1, pt2 = joint
                    if (person[pt1,0] == 0 and person[pt1,1] == 0) or (person[pt2,0] == 0 and person[pt2,1] == 0) :
                        continue
                    cv2.line(frame, (person[pt1,0], person[pt1,1]), (person[pt2,0], person[pt2,1]), self.color_joints, 5)
                    #cv2.line(image, (x1, y1), (x2, y2), (0,255,0), lineThickness)
                for i in range(person.shape[0]):
                    x, y, z = person[i]
                    if x == 0 and y == 0:
                        continue
                    cv2.circle(frame, (x, y), self.points_radius, self.color_points, -1)
            return frame