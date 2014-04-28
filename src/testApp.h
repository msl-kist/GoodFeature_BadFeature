#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxUI.h"

#include "parameters.h"
#include "brisk/brisk.h"
#include "freak/freak.h"

#include <map>

using namespace std;

#define NUMBER_OF_DB_IMAGES	16		// 최종은 16
#define SCALES {1.0}

#define MATCH_SCORE_THRESHOLD 4
#define NUMBER_OF_BUFFER_FRAMES	3


struct MatchResult
{
	vector<cv::DMatch>	matches_brisk, matches_brisk_opp, matches_brisk_common;
	vector<cv::DMatch>	matches_freak, matches_freak_opp, matches_freak_common;
	vector<cv::DMatch>	matches_hybrid, matches_hybrid_opp, matches_hybrid_common;

	float match_score;
	bool matched;
};

class testApp : public ofBaseApp{

	public:
		void setup();

		void LoadTrain();

		void update();
		void draw();
		void exit();

		void keyPressed  (int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		ofTrueTypeFont	font;

		// Camera
		//------------------------------
		ofVideoGrabber		camera;
		
		// we will have a dynamic number of images, based on the content of a directory:
		ofDirectory dir;

		// 현재 영상 포인터
		ofImage *	imageCurrentQuery;
		ofImage		imageTrain;

		bool processDone;
		bool bDrawLine;

		// Matching
		//------------------------------
		cv::BFMatcher *matcher; // recent matcher

		// BRISK 저자 코드
		//------------------------------
		cv::Ptr<cv::FeatureDetector> featureDetector;
		cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;

		vector<cv::KeyPoint>			keyPointsTrain;
		vector<cv::KeyPoint>			keyPointsQuery;
		cv::Mat							descriptorsTrain;
		cv::Mat							descriptorsQuery;

		vector<cv::DMatch>				matches;
		Mat outliers;

		// 논문용
		vector<cv::KeyPoint>			goodKeypointsTrain;
		vector<cv::KeyPoint>			badKeypointsTrain;
		cv::Mat							goodDescriptorsTrain;
		cv::Mat							badDescriptorsTrain;

		int								numberOfGoodKeypoints;
		

		// 그림 그리는 용
		//--------------------------------------------------------------
		vector<ofPoint>					matchedTraining;
		vector<ofPoint>					matchedQuery;


		void computeDescriptors(ofImage * image, vector<cv::KeyPoint> * keyPoints, cv::Mat * descriptors);


		float MatchScoreThreshold;
		bool matched;
		int	matchedIndex;

		// Drawing Homography
		//==============================
		vector<cv::Point2f> convertOF2CV(vector<ofPoint> * source);
		vector<ofPoint> convertCV2OF(vector<cv::Point2f> * source);

		bool niceHomography(const Mat * H);

		void GetROI(ofImage * imageTrain,  vector<ofPoint> * matchedQuery, vector<ofPoint> * matchedTraining);
		ofPoint	roi[4];
		float ransacDistThresh;

		int counterInliers;
		int counterOutliers;

		int goodInliers;
		int badInliers;

		ofxUICanvas *gui;
		void setupGUI();

		bool bDrawOutliers;
};
