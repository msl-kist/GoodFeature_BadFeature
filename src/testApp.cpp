#include "testApp.h"

#define IsGenuine	((indexTrain==indexScale) ? true : false)

// GUI
#define	_GUI_WIDTH	300
#define	_GUI_HEIGHT	1080
#define	_SPACER_THICK	1
float xInit = OFX_UI_GLOBAL_WIDGET_SPACING * 2;

//--------------------------------------------------------------
void testApp::setup(){
	ofSetFrameRate(30);
	font.loadFont("verdana.TTF", 32);

	// 카메라
	//------------------------------
	camera.setVerbose(true);
	camera.initGrabber(640,480);

	// 매칭 초기화
	//------------------------------
	matcher = new cv::BFMatcher(cv::NORM_HAMMING, false);

	// BRISK 저자 코드
	//------------------------------
	featureDetector = new cv::GridAdaptedFeatureDetector(new cv::BriskFeatureDetector(80, 5));
	descriptorExtractor = new cv::BriskDescriptorExtractor();

	// DB 영상 입력 및 키포인트/디스크립터 계산
	//------------------------------
	LoadTrain();


	// 현재 이미지 세팅
	imageCurrentQuery	= new	ofImage();
	imageCurrentQuery->allocate(640,480, OF_IMAGE_COLOR);

	processDone = false;
	bDrawLine = true;

	ransacDistThresh = 5.0;

	setupGUI();
}

void testApp::LoadTrain()
{
	// Image Load
	//--------------------------------------------------------------
	imageTrain.loadImage("train.jpg");

	FileStorage fsG("GoodFeature.xml",FileStorage::READ);
	FileStorage fsB("BadFeature.xml",FileStorage::READ);

	// Error Handling
	//--------------------------------------------------------------
	if(!fsG.isOpened() || !fsB.isOpened())
	{
		cout << "FILE NOT FOUND";
		return;
	}

	// Read
	//--------------------------------------------------------------
	FileNode features = fsG["Keypoint"];
	read(features, goodKeypointsTrain);

	features = fsB["Keypoint"];
	read(features, badKeypointsTrain);

	fsG["Descriptor"] >> goodDescriptorsTrain;
	fsB["Descriptor"] >> badDescriptorsTrain;

	fsB.release();
	fsG.release();

	numberOfGoodKeypoints = goodKeypointsTrain.size();

	// Merge
	//--------------------------------------------------------------
	keyPointsTrain.resize(goodKeypointsTrain.size() + badKeypointsTrain.size());

	keyPointsTrain.assign(goodKeypointsTrain.begin(), goodKeypointsTrain.end());
	keyPointsTrain.insert(keyPointsTrain.end(), badKeypointsTrain.begin(), badKeypointsTrain.end());

	descriptorsTrain = cv::Mat(cv::Size(goodDescriptorsTrain.cols, goodDescriptorsTrain.rows + badDescriptorsTrain.rows), goodDescriptorsTrain.type());
	
	int index = 0;
	for(int i=0; i<goodDescriptorsTrain.rows; ++i, ++index)
		goodDescriptorsTrain.row(i).copyTo(descriptorsTrain.row(index));
	for(int i=0; i<badDescriptorsTrain.rows; ++i, ++index)
		badDescriptorsTrain.row(i).copyTo(descriptorsTrain.row(index));
}

//--------------------------------------------------------------
void testApp::update(){
	camera.update();

	if(camera.isFrameNew())
	{
		matched = false;

		imageCurrentQuery->setFromPixels(camera.getPixels(),640,480,OF_IMAGE_COLOR);

		// Descriptor Merge 확인
		//==============================================================
		if(false)
		{
			cout << "Good: " << goodDescriptorsTrain.cols << ", " << goodDescriptorsTrain.rows << endl;
			cout << "Bad : " << badDescriptorsTrain.cols << ", " << badDescriptorsTrain.rows << endl;
			cout << "Tot : " << descriptorsTrain.cols << ", " << descriptorsTrain.rows << endl;

			for(int i=0; i<goodDescriptorsTrain.rows; ++i)
				cout << goodDescriptorsTrain.row(i).colRange(cv::Range(0, 3)) << endl;
			cout << goodDescriptorsTrain.row(goodDescriptorsTrain.rows-1) << endl;
			waitKey(0);
			for(int i=0; i<badDescriptorsTrain.rows; ++i)
				cout << badDescriptorsTrain.row(i).colRange(cv::Range(0, 3)) << endl;
			cout << badDescriptorsTrain.row(badDescriptorsTrain.rows-1) << endl;
			waitKey(0);
			for(int i=0; i<descriptorsTrain.rows; ++i)
				cout << descriptorsTrain.row(0).colRange(cv::Range(0, 3)) << endl;
			cout << descriptorsTrain.row(descriptorsTrain.rows-1) << endl;
		}


		// Descriptor 생성
		//==============================================================
		computeDescriptors( imageCurrentQuery,
							&keyPointsQuery,
							&descriptorsQuery);


		// Matching
		//==============================================================
		matcher->match(descriptorsQuery, descriptorsTrain, matches);

		// 매칭 포인트 출력
		//------------------------------
		matchedTraining.clear();
		matchedQuery.clear();

		for(int j=0; j<matches.size(); ++j)
		{
			ofPoint p1;
			p1.x = keyPointsTrain.at(matches.at(j).trainIdx).pt.x;
			p1.y = keyPointsTrain.at(matches.at(j).trainIdx).pt.y;
		
			cv::Point2f p = keyPointsQuery.at( matches.at(j).queryIdx ).pt;
		
			ofPoint p2(p.x, p.y);
		
			matchedTraining.push_back(p1);
			matchedQuery.push_back(p2);
		}

		// 호모그라피 계산
		//------------------------------
		GetROI( &imageTrain, &matchedTraining, &matchedQuery );
	}
}

//--------------------------------------------------------------
void testApp::draw(){
	ofBackground(ofColor::black);

	ofPushMatrix();
	
	if(gui->isVisible())
		ofTranslate(_GUI_WIDTH+10, 0);
	

	imageCurrentQuery->draw(0,0);

	ofSetColor(ofColor::yellow);
	for(int i=0; i<matchedQuery.size(); ++i)
		ofCircle(matchedQuery[i], 3);
	ofSetColor(ofColor::white);


	// Train 이미지
	//--------------------------------------------------------------
	ofPushMatrix();
	ofTranslate(0, 490);
	ofScale(0.6, 0.6);
	imageTrain.draw(0, 0);

	ofSetColor(ofColor::lightGreen);
	for(int i=0; i<goodKeypointsTrain.size(); ++i)
		ofCircle(ofPoint(goodKeypointsTrain[i].pt.x, goodKeypointsTrain[i].pt.y), 4);
	ofSetColor(ofColor::darkOrchid);
	for(int i=0; i<badKeypointsTrain.size(); ++i)
		ofCircle(ofPoint(badKeypointsTrain[i].pt.x, badKeypointsTrain[i].pt.y), 4);
	ofSetColor(ofColor::white);
	ofPopMatrix();

	// XML 읽은 점 출력하여 확인
	//==============================================================
	if(false){
		ofSetColor(ofColor::green);
		for(int i=0; i<goodKeypointsTrain.size(); ++i)
			ofCircle(ofPoint(goodKeypointsTrain[i].pt.x, goodKeypointsTrain[i].pt.y), 5);
		ofSetColor(ofColor::red);
		for(int i=0; i<keyPointsTrain.size(); ++i)
			ofCircle(ofPoint(keyPointsTrain[i].pt.x, keyPointsTrain[i].pt.y), 3);
	}


	// Inlier vs. Outlier
	//--------------------------------------------------------------
	counterOutliers = counterInliers = 0;
	goodInliers = badInliers = 0;

	for(int i=0; i<matchedQuery.size(); ++i)
	{
		if(outliers.at<bool>(i, 0))
		{
			if(bDrawLine)
			{
				if(matches[i].trainIdx < numberOfGoodKeypoints){
					ofSetColor(ofColor::green);
					goodInliers++;
				} else {
					ofSetColor(ofColor::darkGreen);
					badInliers++;
				}
				ofSetLineWidth(1.5);

				ofLine(matchedQuery[i], (matchedTraining[i]*0.6 + ofPoint(0, 490)));
			}

			counterInliers++;
		} else {
			if(bDrawOutliers && bDrawLine)
			{
				ofSetColor(ofColor::gray);
				ofSetLineWidth(0.5);

				ofLine(matchedQuery[i], (matchedTraining[i]*0.6 + ofPoint(0, 490)));
			}
			counterOutliers++;
		}
	}

	ofSetColor(ofColor::white);
	char str[100];
	sprintf(str, "Inliers: %d", counterInliers);
	ofDrawBitmapString(str, 650, 10);
	sprintf(str, "Outliers: %d", counterOutliers);
	ofDrawBitmapString(str, 650, 30);
	
	if(counterInliers > 0)
	{
		sprintf(str, "Good Inliers: %d (%f%%)", goodInliers, (float)goodInliers / (float)counterInliers * 100.0);
		ofDrawBitmapString(str, 650, 50);
		sprintf(str, "BadInliers: %d (%f%%)", badInliers, (float)badInliers / (float)counterInliers * 100.0);
		ofDrawBitmapString(str, 650, 70);
	}

	ofSetColor(ofColor::white);

	ofPopMatrix();	

}

//--------------------------------------------------------------
void testApp::keyPressed(int key)
{
	if(key == 'm')
		gui->toggleVisible();
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}

//--------------------------------------------------------------
void testApp::exit(){

}

void testApp::computeDescriptors(ofImage * image, vector<cv::KeyPoint> * keyPoints, cv::Mat * descriptors)
{
	// ofImage -> OpenCV Mat (BGR)
	//------------------------------
	cv::Mat mat(image->getHeight(), image->getWidth(), CV_8UC3, image->getPixels());
	cv::cvtColor(mat, mat, CV_RGB2BGR);

	// OpenCV Mat(BGR) -> OpenCV Mat(Gray)
	cv::Mat mat_gray(mat.size(), CV_8U);
	cv::cvtColor(mat, mat_gray, CV_BGR2GRAY);

	// 키포인트 찾고 디스크립터 계산
	//------------------------------
	vector<cv::KeyPoint>	kp, kpByBRISK;

	// BRISK 저자 코드
	//------------------------------
	featureDetector->detect(mat_gray, *keyPoints);
	descriptorExtractor->compute(mat_gray, *keyPoints, *descriptors);
}

void testApp::GetROI( ofImage * imageTrain, vector<ofPoint> * matchedQuery, vector<ofPoint> * matchedTraining)
{
	if(matchedQuery->size() < 4)
		return;

	vector<cv::Point2f> pointsQuery = convertOF2CV(matchedQuery), pointsTraining = convertOF2CV(matchedTraining);
	
	Mat H = findHomography(pointsQuery, pointsTraining, RANSAC, ransacDistThresh, outliers);

	vector<Point2f> obj_corners1(4);
	obj_corners1[0] = cvPoint(0,0); 
	obj_corners1[1] = cvPoint( imageTrain->width, 0 );
	obj_corners1[2] = cvPoint( imageTrain->width, imageTrain->height); 
	obj_corners1[3] = cvPoint( 0, imageTrain->height );	

	//Convert Object Corners to Transformed Object Corners Using Homography Matrix Information
	vector<Point2f>				dst_matching_corners(4);
	perspectiveTransform( obj_corners1, dst_matching_corners, H);

	if(!niceHomography(&H))
	{
		matched = false;
		return;
	}

	float ratio = 0.8;
	for(int i=0; i<4; ++i){
		roi[i].x = dst_matching_corners[i].x * ratio + roi[i].x * (1-ratio);
		roi[i].y = dst_matching_corners[i].y * ratio + roi[i].y * (1-ratio);
	}
}

vector<cv::Point2f> testApp::convertOF2CV(vector<ofPoint> * source)
{
	vector<cv::Point2f>	destination;

	for(int i=0; i<source->size(); ++i)
		destination.push_back( cv::Point2f(source->at(i).x, source->at(i).y) );
	
	return destination;
}

vector<ofPoint> testApp::convertCV2OF(vector<cv::Point2f> * source)
{
	vector<ofPoint> destination;
	
	for(int i=0; i<source->size(); ++i)
		destination.push_back( ofPoint(source->at(i).x, source->at(i).y) );

	return destination;
}

#define GetMatrixValue(m, x, y)	(m->at<double>(x, y))
// Returns whether H is a nice homography matrix or not
bool testApp::niceHomography(const Mat * H)
{
	const double det = GetMatrixValue(H, 0, 0) * GetMatrixValue(H, 1, 1) - GetMatrixValue(H, 1, 0) * GetMatrixValue(H, 0, 1);
	if (det < 0)
		return false;

	const double N1 = sqrt(GetMatrixValue(H, 0, 0) * GetMatrixValue(H, 0, 0) + GetMatrixValue(H, 1, 0) * GetMatrixValue(H, 1, 0));
	if (N1 > 4 || N1 < 0.1)
		return false;

	const double N2 = sqrt(GetMatrixValue(H, 0, 1) * GetMatrixValue(H, 0, 1) + GetMatrixValue(H, 1, 1) * GetMatrixValue(H, 1, 1));
	if (N2 > 4 || N2 < 0.1)
		return false;

	const double N3 = sqrt(GetMatrixValue(H, 2, 0) * GetMatrixValue(H, 2, 0) + GetMatrixValue(H, 2, 1) * GetMatrixValue(H, 2, 1));
	if (N3 > 0.002)
		return false;

	return true;
}



void testApp::setupGUI()
{
	float dim = 16;
	gui = new ofxUICanvas(0, 0, _GUI_WIDTH, _GUI_HEIGHT);
	gui->setColorBack(ofColor::darkSlateGray);

	gui->addSpacer(_GUI_WIDTH-10, _SPACER_THICK);
	gui->addWidgetDown(new ofxUILabel("DB Filtering", OFX_UI_FONT_LARGE)); 

	gui->addSpacer(_GUI_WIDTH-10, _SPACER_THICK);
	gui->addSlider("RANSAC THRESHOLD", 0, 20, &ransacDistThresh, _GUI_WIDTH - xInit, dim);

	gui->addSpacer(_GUI_WIDTH-10, _SPACER_THICK);
	gui->addToggle("DRAW LINES", &bDrawLine, dim, dim);
	gui->addToggle("DRAW OUTLIERS", &bDrawOutliers, dim, dim);
}