#include <vector>

#include <opencv.hpp>

using namespace std;
using namespace cv;

#ifndef PARAMETER_H

	#define PARAMETER_H

	//#define fastdetector_threshold_A 127 // 200개
	//#define fastdetector_threshold_A 143 // 98개
	//#define fastdetector_threshold_A 175 // 20개
	#define fastdetector_threshold_A 100
	#define fastdetector_threshold_B 70


	//#define briskdetector_threshold_A 100 // 195개   85// 316개
	#define briskdetector_threshold_A  40//120//100//80//130//180 // 60 : default
	#define briskdetector_threshold_B  40//50//50//80//130 //50

	#define brisk_octave 5
	#define brisk_patterScale 1.0f
	#define ReprojThreshold 3

	//#define surfdetector_threshold_A 4500 //Keypoints 205      2800//keypoints 307
	#define surfdetector_threshold_A 10000//4000//10000//6000
	#define surfdetector_threshold_B 5000//4000//5000//1800	

	#define distance_threshold		100//150//80 //200//100
	#define ratio_threshold			0.9

	#define NUM_TEST_FRAME          10
	#define IMGA_NUMBER				3//10
	#define IMGB_NUMBER				10//100


	#define USE_CAM					1
	#define CV_SSSE3				0
	#define PRINT_IMGB_INTENSITY	0
	#define USE_ALL_KEYPOINTS		0
	#define Matching_Method_1		0  // Distance Threshold
	#define Matching_Method_2		0  // Ratio Threshold
	#define Matching_Method_3		0  // Distance Threshold + Ratio Threshold	
	#define Matching_Method_4       0  // Symmetric Matching
	#define Matching_Method_5		0  // Distance Threshold + Symmetric Matching
	#define Matching_Method_6		1  // Distance Threshold + Ratio Threshold + Symmetric Matching
	#define SAVE_INDEX_DISTANCE		0
	#define RECORDING				0
	#define FRAME_SAVE				0
	#define SELECT_PAIRS            0

	#define SHOW_ALL_KEYPOINTS      1
	#define SHOW_INTEREST_KEYPOINTS 0

	#define g_image_width     640//640//640
	#define g_image_height    480//360//480
	#define frame_per_second   15  // or 30   //fps같은 변수가 라이버러리나 인클루드 파일에 포함되어있으면 컴파일이 안됨

	#define angle_B			0
	#define scale_B			1

	#define FAST_DETECTOR_ABLE				0
	#define SURF_DETECTOR_ABLE				0
	#define ORB_DETECTOR_ABLE				0
	#define BRISK_DETECTOR_ABLE				1

	#define FREAK_DESCRIPTOR_ABLE			1
	#define	BRISK_DESCRIPTOR_ABLE			0

	#define FILE_NAME 1


void gcFilterMatches(vector<DMatch> src, vector<DMatch> src_opp, vector<DMatch>& dst, 
	vector<KeyPoint> _keypointsA, vector<KeyPoint> _keypointsB);

void gcMatches2Points(vector<DMatch> matches, vector<Point2f>& _keypointsA_filtered, vector<Point2f>& _keypointsB_filtered,
	vector<KeyPoint> _keypointsA, vector<KeyPoint> _keypointsB);

void gcKeypointsRecover(vector<KeyPoint> src, vector<KeyPoint> &dst, vector<KeyPoint> keypoints_original);

void gcHomography(bool computeHomography, Mat src, Mat &dst, 
	vector<Point2f> keypointsA_filtered, vector<Point2f> keypointsB_filtered, Scalar color);

#endif