import numpy as np
from utils import *


if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
