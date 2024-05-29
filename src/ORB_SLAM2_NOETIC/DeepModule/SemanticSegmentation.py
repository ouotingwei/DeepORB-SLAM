#!/usr/bin/env python3
import socket
import struct
import cv2
import numpy as np
import pickle

import rospy
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import pickle

model = YOLO("/home/wei/DeepORB-SLAM/src/ORB_SLAM2_NOETIC/DeepModule/semantic_weight/bestnamegood.pt")
results = None
results_lock = threading.Lock()

img_h = 480
img_w = 640

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_data = None
        rospy.init_node('image_subscriber', anonymous=True)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

    def image_callback(self, data):
        image_data = self.bridge.imgmsg_to_cv2(data, "bgr8")
        global results
        with results_lock:
            results = model(image_data)  # predict on each frame

    def get_image_data(self):
        return self.image_data

def receive_two_sets_of_keypoints(client_socket):
    # received undistord kp
    num_keypoints1 = struct.unpack('i', client_socket.recv(4))[0]
    keypoints_set1 = []
    for _ in range(num_keypoints1):
        x = struct.unpack('f', client_socket.recv(4))[0]
        y = struct.unpack('f', client_socket.recv(4))[0]

        size = struct.unpack('f', client_socket.recv(4))[0]
        angle = struct.unpack('f', client_socket.recv(4))[0]
        response = struct.unpack('f', client_socket.recv(4))[0]
        octave = struct.unpack('i', client_socket.recv(4))[0]
        class_id = struct.unpack('i', client_socket.recv(4))[0]

        # create KeyPoint 
        kp = cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
        keypoints_set1.append(kp)

    num_keypoints2 = struct.unpack('i', client_socket.recv(4))[0]

    # received distord kp
    keypoints_set2 = []
    for _ in range(num_keypoints2):
        x = struct.unpack('f', client_socket.recv(4))[0]
        y = struct.unpack('f', client_socket.recv(4))[0]

        size = struct.unpack('f', client_socket.recv(4))[0]
        angle = struct.unpack('f', client_socket.recv(4))[0]
        response = struct.unpack('f', client_socket.recv(4))[0]
        octave = struct.unpack('i', client_socket.recv(4))[0]
        class_id = struct.unpack('i', client_socket.recv(4))[0]

        # create KeyPoint 
        kp = cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
        keypoints_set2.append(kp)

    # received descriptor's row, col, and type
    rows = struct.unpack('i', client_socket.recv(4))[0]
    cols = struct.unpack('i', client_socket.recv(4))[0]
    type_ = struct.unpack('i', client_socket.recv(4))[0]

    if type_ == cv2.CV_32F:
        dtype = np.float32
    elif type_ == cv2.CV_8U:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported cv::Mat type: {type_}")

    elem_size = np.dtype(dtype).itemsize
    descriptor_size = rows * cols * elem_size

    # received the descriptors from client
    descriptor_data = bytearray()
    while len(descriptor_data) < descriptor_size:
        packet = client_socket.recv(descriptor_size - len(descriptor_data))
        if not packet:
            break
        descriptor_data.extend(packet)

    # data type -> cv::Mat
    m_descriptors = np.frombuffer(descriptor_data, dtype=dtype).reshape((rows, cols))

    return keypoints_set1, keypoints_set2, m_descriptors


def send_two_sets_of_keypoints(client_socket, keypoints_set1, keypoints_set2, descriptors):
    # quantity & type
    num_keypoints1 = len(keypoints_set1)
    num_keypoints1_bytes = struct.pack('i', num_keypoints1)
    client_socket.send(num_keypoints1_bytes)

    # undistord kp
    for kp in keypoints_set1:
        x_bytes = struct.pack('f', kp.pt[0])
        y_bytes = struct.pack('f', kp.pt[1])
        client_socket.send(x_bytes)
        client_socket.send(y_bytes)

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

    num_keypoints2 = len(keypoints_set2)
    num_keypoints2_bytes = struct.pack('i', num_keypoints2)
    client_socket.send(num_keypoints2_bytes)

    # distord kp
    for kp in keypoints_set2:
        x_bytes = struct.pack('f', kp.pt[0])
        y_bytes = struct.pack('f', kp.pt[1])
        client_socket.send(x_bytes)
        client_socket.send(y_bytes)

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

    # sending descriptor's row, col, and type
    rows, cols = descriptors.shape
    type_ = descriptors.dtype

    # data type -> OpenCV Mat 
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
    client_socket.send(descriptors.tobytes())

def handle_client(client_socket):
    try:
        # waiting for "results" update
        while results is None:
            pass

        with results_lock:
            # Get masks and classes
            masks = results[0].masks.data
            classes = results[0].boxes.cls

            # Find masks corresponding to the person class (class index 0)
            person_class_index = 24  # the person class index
            person_masks = [masks[i] for i in range(len(classes)) if classes[i] == person_class_index]

            combined_person_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            for idx, mask in enumerate(person_masks):
                mask_resized = cv2.resize(mask.cpu().numpy().astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                combined_person_mask = cv2.bitwise_or(combined_person_mask, mask_resized)

        # received data from client
        keypointsUn, keypoints, mDescriptors = receive_two_sets_of_keypoints(client_socket)

        keep_indices = []
        for i in range(len(keypointsUn)):
            kp_u = int(keypoints[i].pt[0])
            kp_v = int(keypoints[i].pt[1])

            if combined_person_mask[kp_v, kp_u] == 0:
                keep_indices.append(i)

        # keep des & kp without person
        filtered_keypointsUn = [keypointsUn[i] for i in keep_indices]
        filtered_keypoints = [keypoints[i] for i in keep_indices]
        filtered_descriptors = mDescriptors[keep_indices]

        # sending data to client
        send_two_sets_of_keypoints(client_socket, filtered_keypointsUn, filtered_keypoints, filtered_descriptors)

    finally:
        # shutdown
        client_socket.close()


def main():
    # create the TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ("127.0.0.1", 8888)
    server_socket.bind(server_address)
    
    # listen
    server_socket.listen(1)
    
    print("Waiting for a connection...")

    # subscriber threads
    image_subscriber = ImageSubscriber()

    while True:
        # waiting for the client
        client_socket, client_address = server_socket.accept()
        #print("Connection from:", client_address)
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

if __name__ == "__main__":
    main()
