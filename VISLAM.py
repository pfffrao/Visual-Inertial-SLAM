from utils import load_data, visualize_trajectory_2d
import numpy as np
import matplotlib.pyplot as plt
import sys


class VISLAM:
    def __init__(self, filename):
        self.t, self.feature, self.linVel, self.rotVel, self.K, self.b, self.cam_T_imu = load_data(file_name=filename)
        self.t = self.t[0,:]
        self.noiseScale = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2], np.float32)
        self.W = np.zeros((6,6), np.float32)
        for i in range(6):
            self.W[i,i] = self.noiseScale[i]
        
        # stereo camera calibration matrix M
        self.M = np.zeros((4,4), np.float32)
        self.M[0:2, 0:3] = self.K[0:2,0:3]
        self.M[2:4, 0:3] = self.K[0:2,0:3]
        self.M[2,3] = -self.K[0,0] * self.b
        print("M:")
        print(self.M)
        self.IMUPoseLoaded = False
        self.calculateIMUPose()

        self.oRr = np.array([[0, -1,  0],
                             [0,  0, -1],
                             [1,  0,  0]], dtype=np.float32)
        
        self.dpiPrototype = np.zeros((4,4), dtype=np.float32)
        self.dpiPrototype[2,2] = 0.0


        self.validfeatures = {}  # a dictionary for valid features, key is time index and value is a list of feature index.
        self.validFeatureLoaded = False
        self.getValidFeatures()
        self.featureLoc = {}  # a dictionary storing ground frame feature locations. Key is feature index, and value is a numpy array of size 3 x N. N is the count of observations on the feature.
        self.P = np.zeros((3,4), dtype=np.float32)  # projection matrix
        self.P[0:3,0:3] = np.identity(3, dtype=np.float32)
        self.V = 1e-2 * np.identity(4, dtype=np.float32)  # observation noise scale
        self.landmark = np.zeros((2, self.feature.shape[1]), dtype=np.float32)
    
    def dataSummary(self):
        print("Length of time frames:\t\t" + str(self.t.shape[0]))
        print("A sample dt:\t\t" + str(self.t[10] - self.t[9]))
        
        print("Length of linVel:\t\t" + str(self.linVel.shape[1]))
        print("A sample linVel:\t\t")
        print(self.linVel[:, 10:15])
        print("Length of rotVel:\t\t" + str(self.rotVel.shape[1]))
        print("A sample rotVel:\t\t")
        print(self.rotVel[:, 10])

        print("Length of features:\t\t" + str(self.feature.shape[2]))
        print("Number of features:\t\t" + str(self.feature.shape[1]))
        print('A sample feature:\t\t')
        print(self.feature[:, 1, 2])

        print("Intrinsic matrix:")
        print(self.K)

        print("stereo camera baseline:" + str(self.b))

    def hatMap(self, x):
        # return the hatmap of x
        return np.array([[0.0,   -x[2], x[1]],
                         [x[2],  0.0,   -x[0]],
                         [-x[1], x[0],  0.0]], dtype=np.float32)

    def piMap(self, q):
        # q should be a 4x1 vector
        if q[2] != 0.0:
            return np.divide(q, q[2])
        else:
            print("piMap: divided by zero!!")
            return q

    def SE3Rodrigues(self, xi):
        # using the SE(3) Rodrigues formula to calculate exp(\hat{xi})
        # xi = [rho, theta]. both rho and theta are 3x1 vectors
        rho = xi[0:3]
        theta = xi[3:6]
        # print("Linear Velocity:")
        # print(rho)
        # print("Rotational Velocity:")
        # print(theta)
        xihat = np.array([[0.0,       -theta[2],   theta[1],   rho[0]],
                          [theta[2],  0.0,        -theta[0],   rho[1]],
                          [-theta[1], theta[0],   0.0,         rho[2]],
                          [0.0,       0.0,        0.0,         0.0   ]], dtype=np.float32)
        thetaNorm = np.abs(np.linalg.norm(theta))
        # check = np.matmul(xihat, np.matmul(xihat, np.matmul(xihat, xihat))) + thetaNorm * thetaNorm * np.matmul(xihat, xihat)
        # print("check should be close to zero:")
        # print(check)
        T = np.identity(4, np.float32)
        T = np.add(T, xihat) + (1.0-np.cos(thetaNorm)) / (thetaNorm * thetaNorm) * np.matmul(xihat, xihat) + \
            (thetaNorm - np.sin(thetaNorm)) / (np.power(thetaNorm, 3)) * np.matmul(xihat, np.matmul(xihat, xihat))
        return T

    def adjSE3Rodrigues(self, xi):
        #  adjoint of SE(3) Rodrigues formula
        rho = xi[0:3]
        theta = xi[3:6]
        thetaNorm = np.linalg.norm(theta)

        thetaHat = self.hatMap(theta)
        rhoHat = self.hatMap(rho)
        
        xihat = np.zeros((6,6), np.float32)
        xihat[0:3, 0:3] = thetaHat
        xihat[3:6,3:6] = thetaHat
        xihat[0:3,3:6] = rhoHat
        
        T = np.identity(6, np.float32)
        # DP:
        xihat2 = np.matmul(xihat, xihat)
        xihat3 = np.matmul(xihat2, xihat)
        xihat4 = np.matmul(xihat3, xihat)

        secondTerm = (3.0*np.sin(thetaNorm) - thetaNorm*np.cos(thetaNorm)) / (2.0 * thetaNorm) * xihat
        thirdTerm = (4.0 - thetaNorm * np.sin(thetaNorm) - 4.0 * np.cos(thetaNorm)) / (2.0 * np.power(thetaNorm, 2)) * xihat2
        fourthTerm = (np.sin(thetaNorm) - thetaNorm * np.cos(thetaNorm)) / (2.0 * np.power(thetaNorm, 3)) * xihat3
        fifthTerm = (2.0 - thetaNorm * np.sin(thetaNorm) - 2.0 * np.cos(thetaNorm))/ (2.0 * np.power(thetaNorm, 4)) * xihat4

        T = T + secondTerm + thirdTerm + fourthTerm + fifthTerm
        return T


    def calculateIMUPose(self):
        # calculate the SE(3) transformation matrix of IMU overtime self.t and store them in self.IMUPose
        for timestamp in range(len(self.t)):
            # print("Calculating " + str(timestamp))
            if(timestamp is 0):
                self.IMUPose = np.zeros((4,4,1), np.float32)
                self.IMUPose[:,:,0] = np.identity(4, np.float32)
                self.InverseIMUPose = self.IMUPose
                self.IIMUPoseCov = np.zeros((6,6,1), np.float32)
                self.IIMUPoseCov[:,:,0] = self.W
            else:
                dt = self.t[timestamp] - self.t[timestamp-1]
                xi = np.array([[self.linVel[0, timestamp-1]],
                               [self.linVel[1, timestamp-1]],
                               [self.linVel[2, timestamp-1]],
                               [self.rotVel[0, timestamp-1]],
                               [self.rotVel[1, timestamp-1]],
                               [self.rotVel[2, timestamp-1]]], np.float32)
                xi = dt * xi
                dT = self.SE3Rodrigues(xi)
                lastT = self.IMUPose[:,:,-1]
                Tnext = np.reshape(np.matmul(lastT, dT), (4,4,1))
                self.IMUPose = np.concatenate((self.IMUPose, Tnext), axis=2)

                # the inverse IMU pose
                # xi = dt * (-xi)
                # dT = self.SE3Rodrigues(xi)
                # lastT = self.InverseIMUPose[:, :, -1]
                # Tnext = np.reshape(np.matmul(dT, lastT), (4,4,1))
                Tnext = Tnext[0:4, 0:4, 0]
                Tnext = np.linalg.inv(Tnext)
                Tnext = np.reshape(Tnext, (4,4,1))
                self.InverseIMUPose = np.concatenate((self.InverseIMUPose, Tnext), axis=2)
                # print("check IMUPose")
                # print(np.matmul(Tnext[0:4,0:4,0], self.IMUPose[0:4,0:4,-1]))
                # covariance update for inverse IMU pose
                lastCov = self.IIMUPoseCov[:, :, -1]
                cov = self.adjSE3Rodrigues(xi)
                covNext = np.matmul(cov, np.matmul(lastCov, cov.T))
                covNext = np.reshape(np.add(covNext, self.W), (6,6,1))
                self.IIMUPoseCov = np.concatenate((self.IIMUPoseCov, covNext), axis = 2)
        self.IMUPoseLoaded = True


    def displayIMUPose(self, t):
        print("IMUPose SE(3) at time: " + str(t))
        print(self.IMUPose[:, :, t])

    def transformToG(self, feature, TF):
        # feature = [ul, vl, ur, vr]
        # solve the equation on page 40 lecture 7 to transform it to ground frame
        # TF should be self.IMUPose
        # return value m should be 4x1
        fsu = self.K[0,0]
        cu = self.K[0,2]
        fsv = self.K[1,1]
        cv = self.K[1,2]
        d = feature[0] - feature[2] 
        ul = feature[0]
        vl = feature[1]

        z = fsu * self.b / d
        x = (ul - cu) * z / fsu
        y = (vl - cv) * z / fsv

        # print("transformToG: x:" + str(x) + ", y:" + str(y) + ", z: " + str(z))
        RTranspose = TF[0:3, 0:3].T
        p = TF[0:3, 3]
        p = np.reshape(p, (3,1))

        opticalLoc = np.array([x,y,z], np.float32)
        # print("optical loc:")
        # print(opticalLoc)
        theR = np.matmul(self.oRr, RTranspose)
        theRinv = theR.T
        # print("theRinv:")
        # print(theRinv)
        temp = np.dot(theRinv, opticalLoc)
        temp = np.reshape(temp, (3,1))
        # print("temp:")
        # print(temp)
        # print("p:")
        # print(p)
        m = np.add(p, temp)
        # m = m[:,0]
        # print(m)
        m = np.reshape(m, (3,1))
        return m

    def isValidFeature(self, f):
        # f should be a 4 x 1 vector
        for idx in range(4):
            if f[idx] != -1.0:
                return True
        return False

    def getValidFeatures(self):
        # get valid features for all time index and update self.validfeatures
        for tidx in range(len(self.t)):
            # print("Updating time index:" + str(tidx))
            self.validfeatures[tidx] = []
            featureN = self.feature.shape[1]
            for fidx in range(featureN):
                f = self.feature[:, fidx, tidx]
                if self.isValidFeature(f):
                    self.validfeatures[tidx].append(fidx)
        self.validFeatureLoaded = True
            # print("Totally "+ str(len(self.validfeatures[tidx])) + " valid features\n")

    def dpiMap(self, q):
        result = np.copy(self.dpiPrototype)
        denom = (q[2] * q[2])
        diagonal = 1 / q[2]
        
        result[0,2] = -q[0] / denom 
        result[1,2] = -q[1] / denom
        result[2,2] = 0.0
        result[3,2] = -q[3] / denom
        
        result[0,0] = diagonal
        result[1,1] = diagonal
        result[3,3] = diagonal
        return result

    def visualMap(self):
    # for each timestamp
    #     for all index in validfeatures
    #           if we have this feature
    #               calculate Htij
    #               
    #           else
    #               use self.transformToG to initialize the feature
        if not self.validFeatureLoaded:
            self.getValidFeatures()
        oTi = self.cam_T_imu
        for tidx in range(len(self.t)):
            print("visualmap timestamp:\t\t" + str(tidx))
            Ut = self.InverseIMUPose[0:4,0:4,tidx]
            coeff = np.matmul(oTi, Ut)
            for fidx in self.validfeatures[tidx]:
                if fidx in self.featureLoc:
                    # calculate H_{t,i,j}
                    prevMu = np.ones((3,1), np.float32)
                    prevMu[0:3, 0] = self.featureLoc[fidx]['mu'][0:3,-1]
                    prevCov = self.featureLoc[fidx]['cov'][0:4,0:4,-1]  # 3x4

                    # calculate H
                    prevMuBar = np.ones((4,1), np.float32)
                    prevMuBar[0:3,0] = prevMu[0:3,0]
                    piTerm = np.matmul(coeff, prevMuBar)
                    # calculate the corresponding observation z for the previous mu
                    prevZ = np.matmul(self.M, self.piMap(piTerm))
                    prevZ = np.reshape(prevZ, (4))

                    dPidq = self.dpiMap(piTerm)
                    lastTerm = np.matmul(coeff, self.P.T)
                    lastTerm = np.matmul(dPidq, lastTerm)
                    Htij = np.matmul(self.M, lastTerm)  # Htij is 4x3
                    
                    # calculate K and update
                    Kt = np.matmul(prevCov, Htij.T)
                    # print("Kt:")
                    # print(Kt)
                    lastTerm = np.matmul(Htij, np.matmul(prevCov, Htij.T)) + self.V
                    if tidx == 975:
                        print("fidx:\t\t" + str(fidx))
                        print("lastTerm")
                        print(lastTerm)
                    lastTermInv = None
                    if np.linalg.cond(lastTerm) < 1/sys.float_info.epsilon:
                        lastTermInv = np.linalg.inv(lastTerm)
                    else:
                        print("Singular lastTerm found.")
                        print("Htij:")
                        print(Htij)
                        print("Kt")
                        print("IMUPose")
                        print(self.IMUPose[:,:, tidx])
                        print("PrevMu:")
                        print(prevMu)
                        break
                    Kt = np.matmul(Kt, lastTermInv)  # 3x4
                    # prevZ = self.feature[:,fidx,tidx-1]
                    Z = self.feature[:, fidx, tidx]
                    # print("Kt:")
                    # print(Kt)
                    # print("Z - prevZ:")
                    # print(Z - prevZ)

                    deltaMu = np.dot(Kt, Z - prevZ)
                    # print("deltaMu:")
                    # print(deltaMu)
                    deltaMu = np.reshape(deltaMu, (3,1))
                    mu = prevMu + deltaMu
                    # print("mu:")
                    # print(mu)
                    mu = np.reshape(mu, (3,1))
                    cov = np.matmul(np.subtract(np.identity(3, np.float32), np.matmul(Kt, Htij)), prevCov)
                    cov = np.reshape(cov, (3,3,1))
                    self.featureLoc[fidx]['mu'] = np.concatenate((self.featureLoc[fidx]['mu'], mu), axis=1)
                    self.featureLoc[fidx]['cov'] = np.concatenate((self.featureLoc[fidx]['cov'], cov), axis=2)
                else:
                    print("initializing fidx:\t\t" + str(fidx))
                    mu = self.transformToG(self.feature[:, fidx, tidx], self.IMUPose[:,:,tidx])
                    cov = 1e-2 * np.identity(3,np.float32)
                    self.featureLoc[fidx] = {
                        'mu': np.reshape(mu, (3,1)),
                        'cov': np.reshape(cov, (3,3,1)),
                    }
            if tidx % 300 == 0:
                self.updateLandmark()
                self.showTrajAndLandMark()
        self.updateLandmark()
        self.showTrajAndLandMark()
                

    def updateLandmark(self):
        for fidx, data in self.featureLoc.items():
            mu = data['mu']
            self.landmark[:, fidx] = mu[0:2,-1]
        

    def showTrajAndLandMark(self):
        fig, ax = visualize_trajectory_2d(self.IMUPose, landmark=self.landmark)
        
        
        


if __name__ == "__main__":
    filename = "./data/0034.npz"
    mySLAM = VISLAM(filename)
    mySLAM.dataSummary()
    mySLAM.visualMap()
    # mySLAM.getValidFeatures()
    # mySLAM.calculateIMUPose()
    # mySLAM.displayIMUPose(100)
    # visualize_trajectory_2d(pose=mySLAM.IMUPose)

