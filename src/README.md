# orbslam2_with_selector
```
author 1:
	TIngWei-Ou 
author 2:
	JhihYang-Long
```
This research intends to improve the localization accuracy of Visual SLAM (Simultaneous Localization and Mapping) via utilizing the large language model (LLM). We propose using such a model, named BERT (Bidirectional Encoder Representations from Transformers), as the feature selector, which may construct the contextual relationships of the feature points and their geometric characteristics, so that satisfactory accuracy can be achieved with less features.

## Model setup


## Data Preprocessing


## ORB-SLAM2 setup
### Third party
- Ubunut : 20.04
- ROS : Noetic
- Pangolin : 0.6
- opencv : 4.2.0
- eigen -> sudo apt-get install libeigen3-dev

### install orb-slam2

#### without ros
```
cd ORB_SLAM2_NOETIC
chmod +x build.sh
./build.sh
```
#### env setup with ros
```
echo 'export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:'"`pwd`/Examples/ROS" >> ~/.bashrc
source ~/.bashrc

```

#### build_ros.sh
```
chmod +x build_ros.sh
./build_ros.sh
```
## run slam with ros
```
roslaunch realsense2_camera rs_camera.launch
rosrun ORB_SLAM2 RGBD Vocabulary/ORBvoc.txt Examples/ROS/ORB_SLAM2/Asus.yaml

```

## Semantic Segmentation
- venv -> source venv/bin/activate
'''
python3 SemanticSegmentation.py
'''

## Feature Selector
### env setup
- nvidia-535
- cuda 12.2
- venv -> source venv/bin/activate

