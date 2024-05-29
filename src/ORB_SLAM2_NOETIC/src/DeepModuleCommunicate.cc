#include <iostream>
#include <string>
#include <cstring> // 添加这一行
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <unordered_map>

#include "DeepModuleCommunicate.h"

// keyPointsSet1 -> undistord
void SendTwoSetsOfKeyPoints(int clientSocket, 
                            const std::vector<cv::KeyPoint>& keyPointsSet1, 
                            const std::vector<cv::KeyPoint>& keyPointsSet2,
                            cv::Mat mDescriptors) {
                                
    int numKeyPoints1 = keyPointsSet1.size();
    int numKeyPoints2 = keyPointsSet2.size();

    send(clientSocket, &numKeyPoints1, sizeof(int), 0);

    for (const auto& kp : keyPointsSet1) {
        float x = kp.pt.x;
        float y = kp.pt.y;
        send(clientSocket, &x, sizeof(float), 0);
        send(clientSocket, &y, sizeof(float), 0);

        float size = kp.size;
        float angle = kp.angle;
        float response = kp.response;
        int octave = kp.octave;
        int class_id = kp.class_id;
        send(clientSocket, &size, sizeof(float), 0);
        send(clientSocket, &angle, sizeof(float), 0);
        send(clientSocket, &response, sizeof(float), 0);
        send(clientSocket, &octave, sizeof(int), 0);
        send(clientSocket, &class_id, sizeof(int), 0);
    }

    send(clientSocket, &numKeyPoints2, sizeof(int), 0);

    for (const auto& kp : keyPointsSet2) {
        float x = kp.pt.x;
        float y = kp.pt.y;
        send(clientSocket, &x, sizeof(float), 0);
        send(clientSocket, &y, sizeof(float), 0);

        float size = kp.size;
        float angle = kp.angle;
        float response = kp.response;
        int octave = kp.octave;
        int class_id = kp.class_id;
        send(clientSocket, &size, sizeof(float), 0);
        send(clientSocket, &angle, sizeof(float), 0);
        send(clientSocket, &response, sizeof(float), 0);
        send(clientSocket, &octave, sizeof(int), 0);
        send(clientSocket, &class_id, sizeof(int), 0);
    }

    int rows = mDescriptors.rows;
    int cols = mDescriptors.cols;
    int type = mDescriptors.type();
    send(clientSocket, &rows, sizeof(int), 0);
    send(clientSocket, &cols, sizeof(int), 0);
    send(clientSocket, &type, sizeof(int), 0);

    if (mDescriptors.isContinuous()) {
        send(clientSocket, mDescriptors.data, rows * cols * mDescriptors.elemSize(), 0);
    } else {
        for (int i = 0; i < rows; ++i) {
            send(clientSocket, mDescriptors.ptr(i), cols * mDescriptors.elemSize(), 0);
        }
    }
}


void receive_two_sets_of_keypoint_data(int clientSocket, 
                                       std::vector<cv::KeyPoint>& keypoints1, 
                                       std::vector<cv::KeyPoint>& keypoints2,
                                       cv::Mat& mDescriptors) {

    int numKeyPoints1;
    recv(clientSocket, &numKeyPoints1, sizeof(int), 0);

    keypoints1.clear(); 
    for (int i = 0; i < numKeyPoints1; ++i) {

        float x, y;
        recv(clientSocket, &x, sizeof(float), 0);
        recv(clientSocket, &y, sizeof(float), 0);

        float size, angle, response;
        int octave, class_id;
        recv(clientSocket, &size, sizeof(float), 0);
        recv(clientSocket, &angle, sizeof(float), 0);
        recv(clientSocket, &response, sizeof(float), 0);
        recv(clientSocket, &octave, sizeof(int), 0);
        recv(clientSocket, &class_id, sizeof(int), 0);

        keypoints1.emplace_back(x, y, size, angle, response, octave, class_id);
    }

    int numKeyPoints2;
    recv(clientSocket, &numKeyPoints2, sizeof(int), 0);

    keypoints2.clear();
    for (int i = 0; i < numKeyPoints2; ++i) {

        float x, y;
        recv(clientSocket, &x, sizeof(float), 0);
        recv(clientSocket, &y, sizeof(float), 0);

        float size, angle, response;
        int octave, class_id;
        recv(clientSocket, &size, sizeof(float), 0);
        recv(clientSocket, &angle, sizeof(float), 0);
        recv(clientSocket, &response, sizeof(float), 0);
        recv(clientSocket, &octave, sizeof(int), 0);
        recv(clientSocket, &class_id, sizeof(int), 0);

        keypoints2.emplace_back(x, y, size, angle, response, octave, class_id);
    }

    int rows, cols, type;
    recv(clientSocket, &rows, sizeof(int), 0);
    recv(clientSocket, &cols, sizeof(int), 0);
    recv(clientSocket, &type, sizeof(int), 0);

    int elemSize = CV_ELEM_SIZE(type);
    int descriptorSize = rows * cols * elemSize;

    std::vector<uchar> descriptorData(descriptorSize);
    recv(clientSocket, descriptorData.data(), descriptorSize, 0);

    mDescriptors = cv::Mat(rows, cols, type, descriptorData.data()).clone();
}
