#ifndef DATA_PREPROCESS_H
#define DATA_PREPROCESS_H

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
#include <vector>
#include <iostream>
#include <numeric>


void model_input(std::vector<cv::KeyPoint>& mvKeysUn, std::vector<cv::KeyPoint>& mvKeys, cv::Mat& mDescriptors, const cv::Mat& imDepth, std::vector<std::vector<float>>& input_data) ;
float kp_depth(const float u_, const float v_, const cv::Mat& imDepth);
float disparity_3d(const float u_, const float v_, const cv::Mat& imDepth);
float normalize(float a, float min_a, float max_a);
                 
void output_decoded(std::vector<cv::KeyPoint>& mvKeysUn, std::vector<cv::KeyPoint>& mvKeys, cv::Mat& mDescriptors, std::vector<float>& output_data);



#endif //DATA_PREPROCESS_H
