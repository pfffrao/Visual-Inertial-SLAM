from utils import load_data
import numpy as np

class VISLAM:
    def __init__(self, filename):
        self.t, self.feature, self.linVel, self.rotVel, self.K, self.b, self.cam_T_imu = load_data(file_name=filename)
        self.t = self.t[0,:]
    
    def dataSummary(self):
        print("Length of time frames:\t\t" + str(self.t.shape[0]))
        print("A sample dt:\t\t" + str(self.t[10] - self.t[9]))
        print("Length of features:\t\t" + str(self.feature.shape[2]))
        print('A sample feature:\t\t')
        print(self.feature[:, 2, 2])
        print("Length of linVel:\t\t" + str(self.linVel.shape[1]))
        print("A sample linVel:\t\t")
        print(self.linVel[:, 10])
        print("Length of rotVel:\t\t" + str(self.rotVel.shape[1]))
        print("A sample rotVel:\t\t")
        print(self.rotVel[:, 10])

    def SE4Rodrigues(self, xi):
        # using the SE(3) Rodrigues formula to calculate exp(\hat{xi})
        # xi = [rho, theta]. both rho and theta are 3x1 vectors
        rho = xi[0:3]
        theta = xi[3:6]
        # print("Linear Velocity:")
        # print(rho)
        # print("Rotational Velocity:")
        # print(theta)
        xihat = np.array([[0.0,       theta[2],   theta[1],   rho[0]],
                          [theta[2],  0.0,        -theta[0],  rho[1]],
                          [-theta[1], theta[0],   0.0,        rho[2]],
                          [0.0,       0.0,        0.0,        0.0   ]], dtype=np.float32)
        thetaNorm = np.linalg.norm(theta)
        T = np.identity(4, np.float32)
        T = np.add(T, xihat) + (1-np.cos(thetaNorm)) / (thetaNorm * thetaNorm) * np.matmul(xihat, xihat) + \
            (thetaNorm - np.sin(thetaNorm)) / (np.power(thetaNorm, 3)) * np.matmul(xihat, np.matmul(xihat, xihat))
        return T

    def calculateIMUPose(self):
        # calculate the SE(3) transformation matrix of IMU overtime self.t and store them in self.IMUPose
        for timestamp in range(10):
            if(timestamp is 0):
                self.IMUPose = np.zeros((4,4,1), np.float32)
                self.IMUPose[:,:,0] = np.identity(4, np.float32)
            else:
                dt = self.t[timestamp] - self.t[timestamp-1]
                xi = np.array([[self.linVel[0, timestamp-1]],
                               [self.linVel[1, timestamp-1]],
                               [self.linVel[2, timestamp-1]],
                               [self.rotVel[0, timestamp-1]],
                               [self.rotVel[1, timestamp-1]],
                               [self.rotVel[2, timestamp-1]]], np.float32)
                xi = dt * xi
                dT = self.SE4Rodrigues(xi)
                lastT = self.IMUPose[:,:,-1]
                Tnext = np.reshape(np.matmul(lastT, dT), (4,4,1))
                self.IMUPose = np.concatenate((self.IMUPose, Tnext), axis=2)

    def displayIMUPose(self, t):
        print("IMUPose SE(4) at time: " + str(t))
        print(self.IMUPose[:, :, t])


if __name__ == "__main__":
    filename = "./data/0027.npz"
    mySLAM = VISLAM(filename)
    mySLAM.dataSummary()
    mySLAM.calculateIMUPose()
    mySLAM.displayIMUPose(1)

