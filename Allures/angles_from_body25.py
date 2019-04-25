import numpy as np

class Geometry_Body25(object):

    instance = None

    def __new__(self):
        if not Geometry_Body25.instance:
            Geometry_Body25.instance = Geometry_Body25.__Geometry_Body25()
        return Geometry_Body25.instance

    def __getattr__(self, name):
        return getattr(Geometry_Body25.instance, name)

    def __setattr__(self, name):
        return setattr(Geometry_Body25.instance, name)

    class __Geometry_Body25():

        #vect1 : vector, vect2 : vector
        def getAngle(self,vect1,vect2):
            if np.linalg.norm(vect1) == 0 or np.linalg.norm(vect2) == 0:
                return 0.

            if 0 < np.abs(np.dot(vect1,vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2)))-1 < 1e-4:
                return np.arccos(int(np.dot(vect1,vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2))))*180/np.pi

            if np.isnan(np.arccos(np.dot(vect1,vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2)))*180/np.pi):
                return 0.

            return np.arccos(np.dot(vect1,vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2)))*180/np.pi

        #points : (25,3)
        def getAngles(self,points):
            vect_32 = points[2,:]-points[3,:]
            vect_34 = points[4,:]-points[3,:]
            alpha_1 = self.getAngle(vect_32,vect_34) if np.asarray(points[2,:]).sum() != 0 and np.asarray(points[3,:]).sum() != 0 and np.asarray(points[4,:]).sum() != 0 else 0.

            vect_65 = points[5,:]-points[6,:]
            vect_67 = points[7,:]-points[6,:]
            alpha_2 = self.getAngle(vect_65,vect_67) if np.asarray(points[5,:]).sum() != 0 and np.asarray(points[6,:]).sum() != 0 and np.asarray(points[7,:]).sum() != 0 else 0.

            vect_109 = points[9,:]-points[10,:]
            vect_1011 = points[11,:]-points[10,:]
            alpha_3 = self.getAngle(vect_109,vect_1011) if np.asarray(points[9,:]).sum() != 0 and np.asarray(points[10,:]).sum() != 0 and np.asarray(points[11,:]).sum() != 0 else 0.

            vect_1312 = points[12,:]-points[13,:]
            vect_1314 = points[14,:]-points[13,:]
            alpha_4 = self.getAngle(vect_1312,vect_1314) if np.asarray(points[12,:]).sum() != 0 and np.asarray(points[13,:]).sum() != 0 and np.asarray(points[14,:]).sum() != 0 else 0.

            vect_1122 = points[22,:]-points[11,:]
            vect_1110 = points[10,:]-points[11,:]
            alpha_5 = self.getAngle(vect_1122,vect_1110) if np.asarray(points[10,:]).sum() != 0 and np.asarray(points[11,:]).sum() != 0 and np.asarray(points[22,:]).sum() != 0 else 0.

            vect_1413 = points[13,:]-points[14,:]
            vect_1419 = points[19,:]-points[14,:]
            alpha_6 = self.getAngle(vect_1413,vect_1419) if np.asarray(points[13,:]).sum() != 0 and np.asarray(points[14,:]).sum() != 0 and np.asarray(points[19,:]).sum() != 0 else 0.

            vect_23 = points[3,:]-points[2,:]
            vect_18 = points[8,:]-points[1,:]
            alpha_7 = self.getAngle(vect_23,vect_18) if np.asarray(points[1,:]).sum() != 0 and np.asarray(points[2,:]).sum() != 0 and np.asarray(points[3,:]).sum() != 0 and np.asarray(points[8,:]).sum() != 0 else 0.

            vect_56 = points[6,:]-points[5,:]
            vect_18 = points[8,:]-points[1,:]
            alpha_8 = self.getAngle(vect_56,vect_18) if np.asarray(points[1,:]).sum() != 0 and np.asarray(points[5,:]).sum() != 0 and np.asarray(points[6,:]).sum() != 0 and np.asarray(points[8,:]).sum() != 0 else 0.

            vect_910 = points[10,:]-points[9,:]
            vect_18 = points[8,:]-points[1,:]
            alpha_9 = self.getAngle(vect_910,vect_18) if np.asarray(points[1,:]).sum() != 0 and np.asarray(points[9,:]).sum() != 0 and np.asarray(points[10,:]).sum() != 0 and np.asarray(points[8,:]).sum() != 0 else 0.

            vect_1213 = points[13,:]-points[12,:]
            vect_18 = points[8,:]-points[1,:]
            alpha_10 = self.getAngle(vect_1213,vect_18) if np.asarray(points[1,:]).sum() != 0 and np.asarray(points[12,:]).sum() != 0 and np.asarray(points[13,:]).sum() != 0 and np.asarray(points[8,:]).sum() != 0 else 0.

            return alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8, alpha_9, alpha_10

        #pt1 : vector, pt2 : vector
        def getDistance(self,pt1,pt2,axis=-1):
            sum_cum = 0
            if len(pt1) != len(pt2) or len(pt1) == 0:
                return 0
            if axis != -1:
                return np.abs(pt1[axis]-pt2[axis])
            for i in range(len(pt1)):
                sum_cum += (pt2[i]-pt1[i])**2
            return np.sqrt(sum_cum)

        #points : (25,3)
        def getDistances(self,points):
            dist_1 = self.getDistance(points[10,:],points[13,:]) if np.asarray(points[10,:]).sum() != 0 and np.asarray(points[13,:]).sum() != 0 else 0.
            dist_2 = self.getDistance(points[11,:],points[14,:]) if np.asarray(points[11,:]).sum() != 0 and np.asarray(points[14,:]).sum() != 0 else 0.
            dist_y_1 = self.getDistance(points[9,:],points[11,:],axis=1) if np.asarray(points[9,:]).sum() != 0 and np.asarray(points[11,:]).sum() != 0 else 0.
            dist_y_2 = self.getDistance(points[12,:],points[14,:],axis=1) if np.asarray(points[12,:]).sum() != 0 and np.asarray(points[14,:]).sum() != 0 else 0.
            return dist_1, dist_2, dist_y_1, dist_y_2