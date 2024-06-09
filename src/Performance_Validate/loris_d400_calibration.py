'''
[ 9.9792816252667338e-03, 6.5348103708624539e-03, 9.9992885256485176e-01, 
  2.2648368490900000e-01, -9.9982014658446139e-01, 1.6192923276330706e-02,
  9.8723715283343672e-03, -5.1141940356500000e-02, -1.6127257115523985e-02,
 -9.9984753112121250e-01, 6.6952288046080444e-03, 9.1600000000000004e-01,
  0., 0., 0., 1. ]
'''

import numpy as np
import csv
from scipy.spatial.transform import Rotation as R

def loris_calibration(T_cam, P_robot):
    T_cal = T_cam @ P_robot
    return T_cal

def change_trajectory(path, T_cam):
    trajectory = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            translation_data = [float(row[i]) for i in range(1, 4)]
            quaternion_data = [float(row[i]) for i in range(4, 8)]

            rotation = R.from_quat(quaternion_data)
            rotation_matrix = rotation.as_matrix()

            P_robot = np.eye(4) 
            P_robot[:3, :3] = rotation_matrix
            P_robot[:3, 3] = translation_data

            trajectory_row = loris_calibration(T_cam, P_robot)

            rotation = trajectory_row[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()
            translation_vector = trajectory_row[:3, 3]

            result_matrix = np.zeros(7)
            result_matrix[:3] = translation_vector
            result_matrix[3:] = quaternion

            trajectory.append(result_matrix)

    return trajectory

def main():
    path = "/home/wei/orbslam_selector/src/Performance_Validate/lab_605/ORB.txt"
    T_cam = np.array([ [9.9792816252667338e-03, 6.5348103708624539e-03, 9.9992885256485176e-01, 2.2648368490900000e-01], 
                       [-9.9982014658446139e-01, 1.6192923276330706e-02,9.8723715283343672e-03, -5.1141940356500000e-02] 
                       [-1.6127257115523985e-02, -9.9984753112121250e-01, 6.6952288046080444e-03, 9.1600000000000004e-01],
                       [0., 0., 0., 1.] ])
    
    change_trajectory(path, T_cam)

if __name__ == "__main__":
    main()
    

    