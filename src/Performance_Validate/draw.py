import csv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

def read_timestamp(path):
    timestamp = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time = float(row[0])
            timestamp.append(time)
    
    return timestamp


def read_gt(path):
    data = []
    
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        
        # 讀取所有行
        for row in reader:
            # 轉換數據類型並交換y和z座標
            new_row = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])]
            data.append(new_row)

    # 以第一行的xyz值為基準，調整所有行的xyz值
    base_x, base_y, base_z = data[0][1], data[0][2], data[0][3]
    
    adjusted_data = []
    for row in data:
        adjusted_row = [
            row[0],  # timestamp
            row[1] - base_x,  # x
            row[2] - base_y,  # y
            row[3] - base_z,  # z
            row[4],  # q1
            row[5],  # q2
            row[6],  # q3
            row[7]   # w
        ]
        adjusted_data.append(adjusted_row)

    rotated_data = []
    for row in adjusted_data:
        rotated_data_row = [ row[0], row[3], row[1], -1*row[2], row[4], row[5], row[6], row[7]]
        rotated_data.append(rotated_data_row)

    return rotated_data


def loris_calibration(T_cam, P_cam):
    T_robot = T_cam @ P_cam
    return T_robot


def change_trajectory(path, T_cam):
    trajectory = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for idx, row in enumerate(reader):
            translation_data = [float(row[i]) for i in range(1, 4)]
            quaternion_data = [float(row[i]) for i in range(4, 8)]

            rotation = R.from_quat(quaternion_data)
            rotation_matrix = rotation.as_matrix()

            Pose_cam = np.eye(4) 
            Pose_cam[:3, :3] = rotation_matrix
            Pose_cam[:3, 3] = translation_data

            # 相机轨迹校正
            camera_trajectory_matrix = loris_calibration(T_cam, Pose_cam)

            rotation = R.from_matrix(camera_trajectory_matrix[:3, :3])
            quaternion = rotation.as_quat()
            translation_vector = camera_trajectory_matrix[:4, 3]

            rotation = camera_trajectory_matrix[:3, :3]
            # 提取 R_00 和 R_10
            R_00 = rotation[0, 0]
            R_10 = rotation[1, 0]

            # 計算 theta
            theta = np.arctan2(R_10, R_00)

            if( idx == 0 ):
                print(theta)

            le_t = math.sqrt(2.2648368490900000e-01 **2 + -5.1141940356500000e-02 **2)

            c2b_x = le_t * math.cos(-1.560815581463662 - theta)
            c2b_y = -5.1141940356500000e-02 * math.sin(-1.560815581463662 - theta)

            result_matrix = np.zeros(7)
            result_matrix[:3] = translation_vector[:3].T
            result_matrix[0] = result_matrix[0]- c2b_x
            result_matrix[1] = result_matrix[1]- c2b_y
            result_matrix[3:] = quaternion

            trajectory.append(result_matrix)

    return trajectory

def create_tum_file(timestamp, data, filename):
    # 檢查時間戳和數據是否長度相同
    if len(timestamp) != len(data):
        raise ValueError("時間戳和數據的長度不一致")

    # 打開文件並寫入數據
    with open(filename, 'w') as file:
        for i in range(len(timestamp)):
            line = f"{timestamp[i]} {' '.join(map(str, data[i]))}"
            file.write(line + '\n')

def create_gt_tum(data, filename):
    with open(filename, 'w') as file:
        for row in data:
            row_str = ' '.join(str(elem) for elem in row)
            file.write(row_str + '\n')

            

def main():
    T_cam_to_base = np.array([ [9.9792816252667338e-03, 6.5348103708624539e-03, 9.9992885256485176e-01, 2.2648368490900000e-01], 
                    [-9.9982014658446139e-01, 1.6192923276330706e-02,9.8723715283343672e-03, -5.1141940356500000e-02], 
                    [-1.6127257115523985e-02, -9.9984753112121250e-01, 6.6952288046080444e-03, 9.1600000000000004e-01],
                    [0., 0., 0., 1.] ])
    # 文件路径
    gt_path = "/home/wei/orbslam_selector/src/Performance_Validate/office/loris_office1-1/ep1232/groundtruth.txt"
    origin_file_path = "/home/wei/orbslam_selector/src/Performance_Validate/office/loris_office1-1/ep1232/orb.txt"
    fsm_file_path = "/home/wei/orbslam_selector/src/Performance_Validate/office/loris_office1-1/ep1232/ep31.txt"

    gt_data = read_gt(gt_path)
    # orb
    origin_data = change_trajectory(origin_file_path, T_cam_to_base)
    origin_ts = read_timestamp(origin_file_path)
    # fsm
    fsm_data = change_trajectory(fsm_file_path, T_cam_to_base)
    fsm_ts = read_timestamp(fsm_file_path)

    create_gt_tum(gt_data, '/home/wei/orbslam_selector/src/Performance_Validate/office/loris_office1-1/ep1232/TUM_gt.txt')
    create_tum_file(origin_ts, origin_data, '/home/wei/orbslam_selector/src/Performance_Validate/office/loris_office1-1/ep1232/TUM_orb.txt')
    create_tum_file(fsm_ts, fsm_data, '/home/wei/orbslam_selector/src/Performance_Validate/office/loris_office1-1/ep1232/TUM_fsm31.txt')

    gt_x_coords = [point[1] for point in gt_data]
    gt_y_coords = [point[2] for point in gt_data]

    origin_x_coords = [point[0] for point in origin_data]
    origin_y_coords = [point[1] for point in origin_data]

    fsm_x = [point[0] for point in fsm_data]
    fsm_y = [point[1] for point in fsm_data]

    '''
    plt.figure()
    plt.plot(gt_x_coords, gt_y_coords, marker='', linestyle='-', color='black',linewidth=1.0, label='GT')
    plt.plot(origin_x_coords, origin_y_coords, marker='', linestyle='-', color='b',linewidth=1.0, label='ORB-SLAM')
    plt.plot(fsm_x, fsm_y, marker='', linestyle='-', color='green',linewidth=1.0, label='ORB-SLAM + FSM(31)')

    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title('Trajectory')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()
    '''


if __name__ == "__main__":
    main()

'''
evo_traj tum TUM_orb.txt TUM_fsm1232.txt TUM_gt.txt --ref=TUM_gt.txt --align_origin --plot_mode=yz -va -p

'''