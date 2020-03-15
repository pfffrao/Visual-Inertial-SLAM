from utils import load_data, visualize_trajectory_2d
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import sys
import pdb


class VISLAM:
    def __init__(self, filename):
        self.t, self.fullFeature, self.linVel, self.rotVel, self.K, self.b, self.cam_T_imu = load_data(file_name=filename)
        self.t = self.t[0,:]
        self.noiseScale = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3], np.float32)
        self.W = np.identity(6, np.float32)
        for i in range(6):
            self.W[i,i] = self.noiseScale[i]
        
        # stereo camera calibration matrix M
        self.M = np.zeros((4,4), np.float32)
        self.M[0:2, 0:3] = np.copy(self.K[0:2,0:3])
        self.M[2:4, 0:3] = np.copy(self.K[0:2,0:3])
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

        self.feature = self.downSample()
        self.validfeatures = {}  # a dictionary for valid features, key is time index and value is a list of feature index.
        self.validFeatureLoaded = False
        self.getValidFeatures()
        self.featureLoc = {}  # a dictionary storing ground frame feature locations. Key is feature index, and value is a numpy array of size 3 x N. N is the count of observations on the feature.
        self.P = np.zeros((3,4), dtype=np.float32)  # projection matrix
        self.P[0:3,0:3] = np.identity(3, dtype=np.float32)
        self.V = 1e-2 * np.identity(4, dtype=np.float32)  # observation noise scale
        self.landmark = np.zeros((2, self.feature.shape[1]), dtype=np.float32)

        self.featureN = self.feature.shape[1]
    
    def downSample(self):
        # choose 1000 feature with equal interval
        interval = 3
        featureN = int(self.fullFeature.shape[1] / interval)
        print("choose features with interval: " + str(interval))
        self.feature = np.zeros((4,featureN,len(self.t)))
        for idx in range(featureN):
            self.feature[:,idx,:] = self.fullFeature[:, interval*idx, :]
        return self.feature

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
        print("Number of features:\t\t" + str(self.featureN))
        print('A sample feature:\t\t')
        print(self.feature[:, 1, 2])

        print("Intrinsic matrix:")
        print(self.K)

        print("stereo camera baseline:" + str(self.b))

        print("extrinsic matrix:")
        print(self.cam_T_imu)

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
        xihat[0:3, 0:3] = np.copy(thetaHat)
        xihat[3:6,3:6] = np.copy(thetaHat)
        xihat[0:3,3:6] = np.copy(rhoHat)
        # print("curly Hat:")
        # print(xihat)
        
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
        # print("TF")
        # print(TF)
        # print("d: ")
        # print(d)
        # print("cu")
        # print(cu)
        # print("cv:")
        # print(cv)
        # print("b")
        # print()

        z = fsu * self.b / d
        x = (ul - cu) * z / fsu
        y = (vl - cv) * z / fsv

        # print("transformToG: x:" + str(x) + ", y:" + str(y) + ", z: " + str(z))
        RTranspose = TF[0:3, 0:3].T
        p = np.add(TF[0:3, 3], self.cam_T_imu[:3,3])
        # p = TF[0:3, 3]
        p = np.reshape(p, (3,1))
        # print("transform to G: p is ")
        # print(p)

        opticalLoc = np.array([[x],[y],[z],[1]], np.float32)
        # print("optical loc:")
        # print(opticalLoc)
        # theR = np.matmul(self.oRr, RTranspose)
        # theRinv = np.linalg.inv(theR)
        # temp = np.dot(theRinv, opticalLoc)
        # temp = np.reshape(temp, (3,1))
        # m = np.add(p, temp)
        # # m = m[:,0]
        # # print(m)
        # m = np.reshape(m, (3,1))
        IMULoc = np.matmul(np.linalg.inv(self.cam_T_imu), opticalLoc)
        GlobalCoord = np.matmul(TF, IMULoc)
        return GlobalCoord

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
                    mu = self.transformToG(self.feature[:, fidx, tidx], self.IMUPose[:,:,tidx])[:3]
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
    

    def cirDotMap(self, sbar):
        # sbar should be a (4,1) numpy array
        theMap = np.zeros((4,6), dtype=np.float32)
        theMap[0:3,0:3] = np.identity(3)
        sHat = self.hatMap(sbar[0:3,0])  # input of hatMap should be a (3,) array
        theMap[0:3, 3:6] = -sHat
        return theMap


    def ObservationModel(self, featureLoc, Ut):
        # given a feature location in the global frame, calculate the pixel frame coordinate, Ut is the inverse IMU pose transformation matrix
        
        homoCoord = np.ones((4,1), dtype=np.float32)
        # transform featureLoc to 3x1
        theFeature = np.reshape(featureLoc[:3], (3,1))
        homoCoord[:3] = theFeature[:3]
        # print("ObservationModel: Ut")
        # print(Ut)
        secondTerm = self.piMap(np.matmul(self.cam_T_imu, np.matmul(Ut, homoCoord)))
        # print("second Term:")
        # print(secondTerm)
        pixel = np.matmul(self.M, secondTerm)
        pixel = np.reshape(pixel, (4,))
        return pixel


    def observationTest(self):
        
        
        # IMUPose = np.array([[ 0.00370742,  -0.99999154,  0.00374294,  20.12086741],
        #                     [ 0.9999861,  -0.00371402,  0.00176665, 5.01624031],
        #                     [-0.00374947, -0.00175274,  0.9999915,  -0.06109919],
        #                     [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)
        IMUPose = np.identity(4,dtype=np.float32)
        Ut = np.linalg.inv(IMUPose)
        print("=================== observation test ================")
        for idx in range(10):
            theFeature = np.copy(self.feature[:, idx , 0])
            print("idx: \t" + str(idx))
            print("The feature is:")
            print(theFeature)
            featureInG = self.transformToG(theFeature, IMUPose)  # 3x1
            featurePix = self.ObservationModel(featureInG, Ut)  # 
            # againInG = self.transformToG(featurePix, IMUPose)
            print("feature in Global:")
            print(featureInG)
            # print("Second Time in Global:")
            # print(againInG)
            # print("Reconstructed pixel:")
            # print(featurePix)
            # print("IMUPose:")
            # print(IMUPose)
            # print("Inverse IMUPose:")
            # print(Ut)

            print(IMUPose * Ut)
            print("Difference: ")
            print(np.subtract(theFeature, featurePix))

    def curlyHatMap(self, xi):
        # map a 6x1 vector to a 6x6 hat matrix
        curH = np.zeros((6,6), dtype=np.float32)
        wHat = self.hatMap(xi[:3])
        vHat = self.hatMap(xi[3:6])
        curH[0:3, 0:3] = wHat
        curH[3:6, 3:6] = wHat
        curH[0:3, 3:6] = vHat
        return curH

    def SE3Hat(self, xi):
        # xi should be a 6x1 vector
        rho = np.copy(xi[0:3, 0])
        theta = np.copy(xi[3:6, 0]) 
        xihat = np.array([[0.0,       -theta[2],   theta[1],   rho[0]],
                          [theta[2],  0.0,        -theta[0],   rho[1]],
                          [-theta[1], theta[0],   0.0,         rho[2]],
                          [0.0,       0.0,        0.0,         0.0   ]], dtype=np.float32)
        return xihat
    def SE3AntiHat(self, aHatMap):
        theRes = np.zeros((6,1), dtype=np.float32)
        theRes[0,0] = aHatMap[0,3]
        theRes[1,0] = aHatMap[1,3]
        theRes[2,0] = aHatMap[2,3]
        theRes[3,0] = aHatMap[2,1]
        theRes[4,0] = aHatMap[0,2]
        theRes[5,0] = aHatMap[1,0]
        return theRes

    def cirDotMapTest(self):
        sbar = np.array([[1],[2],[3],[4]])
        cirdot = self.cirDotMap(sbar)
        print("cirdot:")
        print(cirdot)

    def dpiMapTest(self):
        pi = np.array([[1],[2],[3],[4]])
        dpi = self.dpiMap(pi)
        print("dpiMap result:")
        print(dpi)
    
    def SE3HatTest(self):
        pi = np.array([[1],[2],[3],[4],[5],[6]])
        theHat = self.SE3Hat(pi)
        print("se3Hat result:")
        print(theHat)

    def jointEKF(self):
        self.state = np.zeros((3*self.featureN+6,1), dtype=np.float32)
        self.updCov = np.identity(3*self.featureN+6, dtype=np.float32)
        self.seen = []
        jump = 0
        self.allIMUPose = np.zeros((4,4,1), dtype=np.float32)
        self.W = 1e-1 * np.identity(6, dtype=np.float32)
        self.W[3:6,3:6] = 0.05 * np.identity(3, dtype=np.float32)
        Ut = np.identity(4, dtype=np.float32)
        for tidx in range(len(self.t)):
            print("Timestep: " + str(tidx))
            # prediction
            if tidx == 0:
                continue
            else:
                dt = self.t[tidx] - self.t[tidx-1]
                # dt = 0.0
                xi = np.array([[(self.linVel[0, tidx-1] + self.linVel[0, tidx]) / 2.0],
                               [(self.linVel[1, tidx-1] + self.linVel[1, tidx]) / 2.0],
                               [(self.linVel[2, tidx-1] + self.linVel[2, tidx]) / 2.0],
                               [(self.rotVel[0, tidx-1] + self.rotVel[0, tidx]) / 2.0],
                               [(self.rotVel[1, tidx-1] + self.rotVel[1, tidx]) / 2.0],
                               [(self.rotVel[2, tidx-1] + self.rotVel[2, tidx]) / 2.0]], dtype=np.float32)
                Nt = len(self.validfeatures[tidx])  # features observed in this step
                # Nt = 1
                # inverse IMU pose
                xi = -dt * xi
                print("xi:")
                print(xi)
                # prevState = np.copy(self.state[:6])
                # newState = np.add(prevState, xi)
                
                xiHat = self.SE3Hat(xi)
                dUt = LA.expm(xiHat)
                Ut = np.matmul(dUt, Ut)

                # newHat = LA.logm(Ut)
                # self.state[:6] = self.SE3AntiHat(newHat)
                # stateXi = self.SE3Hat(np.copy(self.state[:6]))
                # stateIMUPose = LA.inv(LA.expm(stateXi))
                # print("Ut:")
                # print(Ut)
                # print("StateIMUPose:")
                # print(stateIMUPose)

                # current inverse IMU pose
                # Ut = LA.expm(xiHat)

                # current forward IMU pose
                IMUPose = LA.inv(Ut)
                self.allIMUPose = np.concatenate((self.allIMUPose, np.reshape(IMUPose, (4,4,1))), axis=2)
                print("IMUPose:")
                print(IMUPose)

                # prediction covariance
                lastCov = np.copy(self.updCov[:6,:6])
                # cov = self.adjSE3Rodrigues(xi)
                curlyHat = self.curlyHatMap(xi)
                # print("Curly Hat:")
                # print(curlyHat)
                cov = LA.expm(curlyHat)
                motionCov = np.matmul(cov, np.matmul(lastCov, cov.T))
                # print("Motion cov:")
                # print(motionCov)

                predCov = np.copy(self.updCov)
                predCov[:6,:6] = np.add(np.copy(motionCov), np.copy(self.W))
                # ============== update ======================
                Hxi = np.zeros((4*Nt,6), dtype=np.float32)
                Hm = np.zeros((4*Nt, 3*self.featureN), dtype=np.float32)

                oTi = self.cam_T_imu
                coeff = np.matmul(oTi, Ut)

                # observed feature pixel loc
                Z = np.zeros((4*Nt,1), dtype=np.float32)
                # predicted feature pixel loc
                predZ = np.copy(Z)

                for idx in range(Nt):
                    # calculate Hxi[4*fidx:(4*fidx+4),0:6]
                    fidx = self.validfeatures[tidx][idx]
                    observed = np.copy(self.feature[:, fidx, tidx])
                    observed[3] = observed[1]
                    Z[4*idx:(4*idx+4), 0] = observed
                    startIdx = 6+fidx*3
                    # if no previous observations, use current one as previous
                    if self.state[startIdx, 0] == 0.0:
                        # print("initializing feature " + str(fidx))
                        self.seen.append(fidx)
                        featureInG = self.transformToG(observed, IMUPose)[:3]
                        self.state[startIdx:startIdx+3] = featureInG
                        # predCov[6+fidx*3:9+fidx*3, 6+fidx*3:9+fidx*3] = np.identity(3, dtype=np.float32)
                        print("feature observed in G:")
                        print(featureInG)

                    prevFeatureLoc = np.copy(self.state[startIdx:startIdx+3])
                    # ===== Hxi =====
                    prevFeatureLocBar = np.ones((4,1), dtype=np.float32)
                    prevFeatureLocBar[0:3] = prevFeatureLoc
                    
                    piTerm = np.matmul(coeff, prevFeatureLocBar)
                    dpidq = self.dpiMap(piTerm)
                    firstTerm = np.matmul(np.matmul(self.M, dpidq), oTi)
                    # dpidq = self.dpiMap(np.matmul(coeff, prevFeatureLocBar))
                    lastTerm = np.copy(self.cirDotMap(np.matmul(Ut, prevFeatureLocBar)))
                    Hi = np.matmul(firstTerm, lastTerm)
                    Hxi[4*idx:(4*idx+4),:] = Hi
                    # print(Hi)

                    # calculate Hm
                    lastTerm = np.matmul(Ut, self.P.T)
                    Htij = np.matmul(firstTerm, lastTerm)
                    Hm[4*idx:4*idx+4, 3*fidx:3*fidx+3] = Htij

                    # update predicted Z according to previous
                    predZ[4*idx:(4*idx+4),0] = self.ObservationModel(prevFeatureLoc, Ut)
                    # predZ[4*idx:(4*idx+4)] = np.matmul(self.M, self.piMap(piTerm))
                
                print("feature Jacobian complete.")
                innovation = np.subtract(Z, predZ)
                # print("Z:")
                # print(Z)
                # print("predZ:")
                # print(predZ)
                print("Max innovation")
                print(np.amax(innovation))
                print("min innvoation:")
                print(np.amin(innovation))

                theH = np.concatenate((Hxi, Hm), axis=1)  # 4Nt x (3Nt+6)
                V = np.identity(4*Nt, dtype=np.float32)
                
                # alternative way to derive K
                K = np.matmul(predCov, theH.T)
                lastTerm = np.matmul(theH, np.matmul(predCov, theH.T))
                lastTerm = np.add(lastTerm, V)
                scaler = np.amax(lastTerm)
                lastTerm = np.divide(lastTerm, scaler)
                theInv = np.identity(4*Nt, dtype=np.float32)
                if np.linalg.cond(lastTerm) < 1/sys.float_info.epsilon:
                    try:
                        theInv = np.linalg.inv(lastTerm)
                    except Exception as e:
                        jump = jump + 1
                        continue
                else:
                    print("Singular lastTerm found.")
                    jump = jump + 1
                    print(lastTerm)
                    continue
                K = np.matmul(K, theInv)
                K = np.divide(K, scaler)

                # S = np.matmul(theH, np.matmul(predCov, theH.T))  # 4Nt x 4Nt
                # S = np.add(S, V)
                # C = np.matmul(predCov, theH.T)  # 3Nt+6 x 4Nt -> 3M+6 x 4Nt
                # Sinv = np.identity(4*Nt,dtype=np.float32)
                # if np.linalg.cond(S) < 1/sys.float_info.epsilon:
                #     try:
                #         Sinv = np.linalg.inv(S)
                #     except Exception as e:
                #         jump = jump + 1
                #         continue
                # else:
                #     print("Singular S found.")
                #     jump = jump + 1
                #     print(S)
                #     continue
                # K = np.matmul(C, Sinv)  # -> 3M+6 x 4Nt
                print()
                print("Kalman Max:")
                print(np.amax(K))
                print("Kalman Min:")
                print(np.amin(K))
                

                delta = np.matmul(K, innovation)
                print("Delta[:6]")
                print(delta[:6])
                print("max Delta")
                print(np.amax(delta))
                print("min Delta")
                print(np.amin(delta))
                delta = np.reshape(delta, (3*self.featureN+6,1))
  
                predMu = np.copy(self.state)
                updateMu = np.add(predMu, delta)  # 3M+6 x 1

                # print(updateMu)
                # index variables to help extract 3Nt+6 x 3Nt submatrix from all 1000+ features
                # rowIdx = np.zeros((3*Nt+6,1), dtype=np.int16)
                # colIdx = np.zeros((1, 3*Nt+6), dtype=np.int16)
                # rowIdx[:6,0] = range(6)
                # colIdx[0,:6] = range(6)
                
                # for idx in range(Nt):
                #     # calculate Hxi[4*fidx:(4*fidx+4),0:6]
                #     fidx = self.validfeatures[tidx][idx] 
                #     startIdx = fidx*3 + 6
                #     rowStart = 3*idx + 6
                #     rowIdx[rowStart:(rowStart+3), 0] = range(startIdx,startIdx+3)
                #     colIdx[0, rowStart:(rowStart+3)] = range(startIdx,startIdx+3)
                
                updateCov = np.subtract(np.identity(3*self.featureN+6, dtype=np.float32), np.matmul(K, theH))
                updateCov = np.matmul(updateCov, predCov)
                #np.subtract(predCov, np.matmul(K, np.matmul(S, K.T)))  # 3M+6 x 3M+6
                # update the state variable
                
                Ut = np.matmul(LA.expm(self.SE3Hat(delta[:6])), Ut)
                self.state = updateMu
                self.updCov = updateCov
                print("update covariance:")
                print(self.updCov)
                if tidx % 300 == 0:
                    print("jumped " + str(jump) + " time steps!")
                    self.landmark = np.reshape(self.state[6:], (3, self.featureN), order='F')
                    print(self.landmark)
                    fig, ax = visualize_trajectory_2d(self.allIMUPose, landmark=self.landmark)
                    
        print("jumped " + str(jump) + " time steps!")
        self.landmark = np.reshape(self.state[6:], (3, self.featureN), order='F')
        print(self.landmark)
        fig, ax = visualize_trajectory_2d(self.allIMUPose, landmark=self.landmark)



if __name__ == "__main__":
    filename = "./data/0027.npz"
    mySLAM = VISLAM(filename)
    # mySLAM.SE3HatTest()
    mySLAM.jointEKF()
    # mySLAM.dataSummary()
    # mySLAM.observationTest()
    # mySLAM.visualMap()
    # mySLAM.cirDotMapTest()
    # mySLAM.dpiMapTest()
    # mySLAM.getValidFeatures()
    # mySLAM.calculateIMUPose()
    # mySLAM.displayIMUPose(100)
    # visualize_trajectory_2d(pose=mySLAM.IMUPose)

