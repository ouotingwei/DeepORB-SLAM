#include "DataPreprocess.h"

void model_input(std::vector<cv::KeyPoint>& mvKeysUn, std::vector<cv::KeyPoint>& mvKeys, cv::Mat& mDescriptors, const cv::Mat& imDepth, std::vector<std::vector<float>>& input_data) {
    // if depth or other data are invalid -> remove & update to mvKeysUn
    std::vector<cv::KeyPoint> mvKeysUnCheck;
    std::vector<cv::KeyPoint> mvKeysCheck;
    cv::Mat mDesCheck;
    std::vector<std::vector<float>> input;
    int N = mvKeys.size();
    for(int i=0; i<N; i++) {
        std::vector<float> input_temp;
        // uv
        float u_ = mvKeys[i].pt.x;
        float v_ = mvKeys[i].pt.y;
        
        /*
        if(mvKeysUn[i].pt.x > 480.0) {
                std:: cout << "aaaaaa : " << mvKeysUn[i].pt.x << std::endl;
            }
            */
        

        // depth
        float depth_ = kp_depth(u_, v_, imDepth);

        // 3d-sparity
        float disparity3d_ = disparity_3d(u_, v_, imDepth);

        // keypoint response
        float response_ = normalize(mvKeys[i].response, 0, 150);

        // keypoint size
        float size_ = mvKeys[i].size / 31;

        // normalize
        u_ = normalize(u_, 19, 828);
        v_ = normalize(v_, 19, 460);
        depth_ = normalize(depth_, 0.376, 4.328);
        disparity3d_ = normalize(disparity3d_, 0, 1);

        // check value
        if( depth_ != 0.0 && depth_ <= 2.5 && disparity3d_ != std::numeric_limits<float>::max() ) {
            //std::cout << disparity3d_ << " " << depth_ << " " << response_ << " " << size_ << std::endl;

            mvKeysUnCheck.push_back(mvKeysUn[i]);
            mvKeysCheck.push_back(mvKeys[i]);
            mDesCheck.push_back(mDescriptors.row(i));

            input_temp.push_back(disparity3d_);
            input_temp.push_back(depth_);
            input_temp.push_back(response_);
            input_temp.push_back(size_);

            input.push_back(input_temp);
        }  
    }
    
    // update
    mvKeysUn = mvKeysUnCheck;
    mvKeys = mvKeysCheck;
    mDescriptors = mDesCheck;
    input_data = input;
}

float normalize(float a, float min_a, float max_a) {
    return (a - min_a) / (max_a - min_a);
}

float kp_depth(const float u_, const float v_, const cv::Mat& imDepth) {
    const int u = static_cast<int>(u_);
    const int v = static_cast<int>(v_);
    float depth = imDepth.at<float>(v, u);

    return depth;
}

float disparity_3d(const float u_, const float v_, const cv::Mat& imDepth) {
    // initialize
    float diff = std::numeric_limits<float>::max();
    float u = u_;
    float v = v_;
    float Z = imDepth.at<float>(v, u);

    if (Z == 0.0) { // 無效深度值
        return diff;
    }

    float total_depth = 0.0f;
    int count = 0;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int u_neigh = static_cast<int>(u) + i;
            int v_neigh = static_cast<int>(v) + j;
            if (u_neigh >= 0 && u_neigh < imDepth.cols && v_neigh >= 0 && v_neigh < imDepth.rows) {
                float depth_value = imDepth.at<float>(v_neigh, u_neigh);
                if (depth_value != -1) { // 無效深度值檢查
                    total_depth += depth_value;
                    count++;
                }
            }
        }
    }

    if (count == 0) {
        return diff;
    }

    // 計算周圍點的平均深度
    float mean_depth = total_depth / count;

    // 計算深度的絕對差值
    diff = std::abs(Z - mean_depth);

    return diff;
}




void output_decoded(std::vector<cv::KeyPoint>& mvKeysUn, std::vector<cv::KeyPoint>& mvKeys, cv::Mat& mDescriptors, std::vector<float>& output_data){
    std::vector<cv::KeyPoint> mvKeysUnCheck;
    std::vector<cv::KeyPoint> mvKeysCheck;
    cv::Mat mDesCheck;

    int N = mvKeys.size();
    for(int i=0; i<N; i++) {
        if(output_data[i] == 1) {

            mvKeysUnCheck.push_back(mvKeysUn[i]);
            mvKeysCheck.push_back(mvKeys[i]);
            mDesCheck.push_back(mDescriptors.row(i));
        }
    }

    mvKeysUn = mvKeysUnCheck;
    mvKeys = mvKeysCheck;
    mDescriptors = mDesCheck;

    std:: cout << "filtered : " << mvKeysUn.size() << std::endl;

    return;
}
