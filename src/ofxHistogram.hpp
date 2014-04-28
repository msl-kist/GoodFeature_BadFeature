#pragma once

#include <opencv.hpp>
#include "ofMain.h"
#include "ofxCvColorImage.h"

class ofxHistogram
{
public:
	const static int dims = 1;

	const static bool uniform = true;

	int histSize;
	float range[2];
	bool accumulate;
	float gap;

	ofxHistogram(int histSize=256, float min=0, float max=256, bool accumulate=false)
		: histSize(histSize), accumulate(accumulate)
	{
		range[0] = min;
		range[1] = max;

		gap = (range[1] - range[0]) / histSize;

		_histogram.create(dims, histSize, CV_32F);
		for(int i=0; i<histSize; ++i)
			_histogram.at<float>(i) = 0;
	}

	void calculate(cv::Mat *image)
	{
		const float* histRange = { range };
		cv::calcHist(image, 1, 0, cv::Mat(), _histogram, 1, &histSize, &histRange, uniform, accumulate);
	}

	void accumulateAValue(float value)
	{
		if(value < range[0] || value > range[1])
		{
			ofLogError("ofxHistogram", "Out of range");
			return;
		}

		for(int i = 0 ; i < histSize-1 ; ++i)
		{
			if(gap * i <= value && gap * (i+1) > value){
				_histogram.at<float>(i) += 1; 
			}
		}
	}

	void draw()
	{
		draw(0, 0, ofGetWidth(), ofGetHeight());
	}

	void draw(int x, int y, int w, int h)
	{
		cv::Mat normalized_histogram;
		cv::normalize(_histogram, normalized_histogram, 0, 1.0, cv::NORM_MINMAX, -1, cv::Mat());

		ofPushMatrix();

		ofTranslate(x, y, 0);

		//ofSetColor(ofColor::black);
		//ofFill();
		//ofRect(0, 0, w, h);

		// y축 역상
		ofTranslate(0, h, 0);
		ofScale(1, -1, 1);

		// Normalized 좌표계
		ofScale(w, h, 1.0);

		ofFill();
		float rect_w = 1.0 / histSize;
		for(int i=0; i<histSize; ++i)
		{
			ofRect(i * rect_w, 0, rect_w, normalized_histogram.at<float>(i));
			
			ofPushStyle();
			ofSetColor(ofColor::white);
			ofDrawBitmapString(ofToString(_histogram.at<float>(i)), i*rect_w, normalized_histogram.at<float>(i));
			ofPopStyle();
		}

		ofPopMatrix();

	}

private:
	cv::Mat _histogram;
};