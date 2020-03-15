import numpy as np
from utils import *
from VISLAM import VISLAM


if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
	mySLAM = VISLAM(filename)

	# (a) IMU Localization via EKF Prediction
	mySLAM.visualMap()  # plot the trajectory and visual map result
	# (b) Landmark Mapping via EKF Update
	mySLAM.visualMap()
	# (c) Visual-Inertial SLAM
	mySLAM.jointEKF()
