#ifndef DEEP_MODULE_COMMUNICATE_H
#define DEEP_MODULE_COMMUNICATE_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

// Semantic Segmentation
void SendTwoSetsOfKeyPoints(int clientSocket, const std::vector<cv::KeyPoint>& keyPointsSet1, const std::vector<cv::KeyPoint>& keyPointsSet2, cv::Mat mDescriptors);

void receive_two_sets_of_keypoint_data(int clientSocket, 
                                       std::vector<cv::KeyPoint>& keypoints1, 
                                       std::vector<cv::KeyPoint>& keypoints2,
                                       cv::Mat& mDescriptors);

// Feature Selection
void send_data_to_FSM(int clientSocket, std::vector<std::vector<float>>& input_data);
std::vector<float> receive_data_from_FSM(int clientSocket);


#endif // DEEP_MODULE_COMMUNICATE_H
