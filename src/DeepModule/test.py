import socket
import struct
import cv2
import numpy as np
import pickle

def receive_two_sets_of_keypoints(client_socket):
    # 接收第一组特征点数量
    num_keypoints1 = struct.unpack('i', client_socket.recv(4))[0]

    # 接收第一组特征点数据
    keypoints_set1 = []
    for _ in range(num_keypoints1):
        # 接收特征点位置
        x = struct.unpack('f', client_socket.recv(4))[0]
        y = struct.unpack('f', client_socket.recv(4))[0]

        # 接收特征点其他属性
        size = struct.unpack('f', client_socket.recv(4))[0]
        angle = struct.unpack('f', client_socket.recv(4))[0]
        response = struct.unpack('f', client_socket.recv(4))[0]
        octave = struct.unpack('i', client_socket.recv(4))[0]
        class_id = struct.unpack('i', client_socket.recv(4))[0]

        # 创建 KeyPoint 对象并添加到列表中
        kp = cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
        keypoints_set1.append(kp)

    # 接收第二组特征点数量
    num_keypoints2 = struct.unpack('i', client_socket.recv(4))[0]

    # 接收第二组特征点数据
    keypoints_set2 = []
    for _ in range(num_keypoints2):
        # 接收特征点位置
        x = struct.unpack('f', client_socket.recv(4))[0]
        y = struct.unpack('f', client_socket.recv(4))[0]

        # 接收特征点其他属性
        size = struct.unpack('f', client_socket.recv(4))[0]
        angle = struct.unpack('f', client_socket.recv(4))[0]
        response = struct.unpack('f', client_socket.recv(4))[0]
        octave = struct.unpack('i', client_socket.recv(4))[0]
        class_id = struct.unpack('i', client_socket.recv(4))[0]

        # 创建 KeyPoint 对象并添加到列表中
        kp = cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
        keypoints_set2.append(kp)

    # 接收描述子的行数、列数和类型
    rows = struct.unpack('i', client_socket.recv(4))[0]
    cols = struct.unpack('i', client_socket.recv(4))[0]
    type_ = struct.unpack('i', client_socket.recv(4))[0]

    # 确定描述子数据的元素大小
    if type_ == cv2.CV_32F:
        dtype = np.float32
    elif type_ == cv2.CV_8U:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported cv::Mat type: {type_}")

    elem_size = np.dtype(dtype).itemsize
    descriptor_size = rows * cols * elem_size

    # 接收描述子数据
    descriptor_data = bytearray()
    while len(descriptor_data) < descriptor_size:
        packet = client_socket.recv(descriptor_size - len(descriptor_data))
        if not packet:
            break
        descriptor_data.extend(packet)

    # 将接收到的数据转换为 cv::Mat
    m_descriptors = np.frombuffer(descriptor_data, dtype=dtype).reshape((rows, cols))

    return keypoints_set1, keypoints_set2, m_descriptors


def send_two_sets_of_keypoints(client_socket, keypoints_set1, keypoints_set2, descriptors):
    # 发送第一组特征点数量
    num_keypoints1 = len(keypoints_set1)
    num_keypoints1_bytes = struct.pack('i', num_keypoints1)
    client_socket.send(num_keypoints1_bytes)

    # 发送第一组特征点数据
    for kp in keypoints_set1:
        # 发送特征点位置
        x_bytes = struct.pack('f', kp.pt[0])
        y_bytes = struct.pack('f', kp.pt[1])
        client_socket.send(x_bytes)
        client_socket.send(y_bytes)

        # 发送特征点其他属性
        size_bytes = struct.pack('f', kp.size)
        angle_bytes = struct.pack('f', kp.angle)
        response_bytes = struct.pack('f', kp.response)
        octave_bytes = struct.pack('i', kp.octave)
        class_id_bytes = struct.pack('i', kp.class_id)

        client_socket.send(size_bytes)
        client_socket.send(angle_bytes)
        client_socket.send(response_bytes)
        client_socket.send(octave_bytes)
        client_socket.send(class_id_bytes)

    # 发送第二组特征点数量
    num_keypoints2 = len(keypoints_set2)
    num_keypoints2_bytes = struct.pack('i', num_keypoints2)
    client_socket.send(num_keypoints2_bytes)

    # 发送第二组特征点数据
    for kp in keypoints_set2:
        # 发送特征点位置
        x_bytes = struct.pack('f', kp.pt[0])
        y_bytes = struct.pack('f', kp.pt[1])
        client_socket.send(x_bytes)
        client_socket.send(y_bytes)

        # 发送特征点其他属性
        size_bytes = struct.pack('f', kp.size)
        angle_bytes = struct.pack('f', kp.angle)
        response_bytes = struct.pack('f', kp.response)
        octave_bytes = struct.pack('i', kp.octave)
        class_id_bytes = struct.pack('i', kp.class_id)

        client_socket.send(size_bytes)
        client_socket.send(angle_bytes)
        client_socket.send(response_bytes)
        client_socket.send(octave_bytes)
        client_socket.send(class_id_bytes)

    # 发送描述子的行数、列数和类型
    rows, cols = descriptors.shape
    type_ = descriptors.dtype

    # 将类型转换为 OpenCV Mat 类型表示
    type_num = 0
    if type_ == np.uint8:
        type_num = cv2.CV_8U
    elif type_ == np.float32:
        type_num = cv2.CV_32F
    else:
        raise ValueError(f"Unsupported descriptor type: {type_}")

    rows_bytes = struct.pack('i', rows)
    cols_bytes = struct.pack('i', cols)
    type_bytes = struct.pack('i', type_num)

    client_socket.send(rows_bytes)
    client_socket.send(cols_bytes)
    client_socket.send(type_bytes)

    # 发送描述子数据
    client_socket.send(descriptors.tobytes())


def handle_client(client_socket):
    try:
        # 接收特征点数据
        keypointsUn, keypoint, descriptor = receive_two_sets_of_keypoints(client_socket)

        N = len(keypointsUn)-1

        # 进行处理...
        del keypointsUn[N]
        del keypoint[N]
        print("Before deletion - descriptor shape:", descriptor.shape)
        descriptor = np.delete(descriptor, N, axis=0)
        print("After deletion - descriptor shape:", descriptor.shape)
        print("KP : ", len(keypointsUn))

        send_two_sets_of_keypoints(client_socket, keypointsUn, keypoint, descriptor)
        

    finally:
        # 关闭客户端连接
        client_socket.close()

def main():
    # 创建 TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 绑定 IP 地址和端口
    server_address = ("127.0.0.1", 8888)  # 替换为实际的服务器端口
    server_socket.bind(server_address)
    
    # 开始监听连接
    server_socket.listen(1)
    
    print("Waiting for a connection...")

    while True:
        # 等待客户端连接
        client_socket, _ = server_socket.accept()
        
        # 在新线程中处理客户端连接
        handle_client(client_socket)

if __name__ == "__main__":
    main()
