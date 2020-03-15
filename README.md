# Visual-Inertial SLAM
## VISLAM Class design
VISLAM is the class that holds the solution to the visual inertial SLAM problem. There are two main functions --- `visualMap()` and `jointEKF()`
### VISLAM::visualMap()
This function solves part a and part b together, outputing a plot of the trajectory and the visual mapping based on the trajectory.

The logic is to deal with the dataset in each timestamp. Since we do not have joint update step, this calculation can be done without using downsampling, i.e. we use all the features in each timestamp.

We established the list of valid features in each time stamp using getValidFeatures() which stores in the `self.validfeatures` all the indices of all valid features in a time stamp. `self.validfeatures[tidx]` is a list that stores the valid features for time stamp tidx.

We use a dictionary `self.featureLoc` to store the dictionary of `mu` and `cov` of each feature, with feature idx `fidx` as its key.

If we haven't seen this feature before, i.e. there is no key `fidx` in `self.featureLoc`. We use the pixel values of this first observation and `self.transformToG()` to get the location of the feature as prior.

If we have seen this feature before, we use the update rule on page 8 of lecture 13 to calculate the Jacobian `Htij` and Kalman gain `Kt`. Use the current observation `Z` and predicted observation `prevZ` to calculate the innovation and update the feature location. The results for the three datasets are as follows:

< img src="Images/Part-a+b/22.png" alt="alt text" style="zoom:50%;" />

< img src="Images/Part-a+b/27.png" alt="alt text" style="zoom:50%;" />

< img src="Images/Part-a+b/34.png" alt="alt text" style="zoom:50%;" />
### VISLAM::jointEKF()
This function solves the problem in part c. It contains a EKF prediction step and a jointly EKF update step to calculate the trajectory and the feature locations.

As the amount of time increased a lot, we performed downsampling. We choose feature using an equal interval. In the code, we chose 3.

In the prediction step, we use the speed input `xi` to calcualte the inverse IMU pose `Ut` and the predition covariance `predCov`. Each step a **0.1I** motion noise is added to the covariance.

In the update step, we use the same logic as that of the part b. If we have not seen the features, we use the first feature reading to initialize the prior. 
Suppose we have totally `M` features and for each time stamp we have `Nt` features observed. The Jacobian is a `4Nt x (3M+6)` matrix which can be divided into two parts `Hxi` --- Jacobian of motion and `Hm` --- Jacobian of feature locations. We use the formula from page 8 and page 19 of lecture 13 to calculate each of them and stack them into one matrix.

Then we use the Jacobian `theH` and the prediction covariance `predCov` to calculate the Kalman Filter.
We used a trick to prevent some numerical issues that may arise when we invert the `theH x predCov x theH + V`, i.e. to divide the matrix by its biggest element and then after inversion divide it again to keep the kalman gain correct.

We used the formula on page 19 of lecture 13 to update the inverse IMU pose as well as the feature locations. The results of each dataset are as follows:

Dataset 22
< img src="Images/Part-C/22.png" alt="alt text" style="zoom:50%;" />

Dataset 22
< img src="Images/Part-C/27.png" alt="alt text" style="zoom:50%;" />

Dataset 22
< img src="Images/Part-C/34.png" alt="alt text" style="zoom:50%;" />

### Helper functions
There are many helper functions: `hatMap()`, `CirDotMap`, `piMap`, `dpiMap` and `ObservationModel`. They helped calculate some important results of EKF update. We also wrote some test for them to ensure they are correct.