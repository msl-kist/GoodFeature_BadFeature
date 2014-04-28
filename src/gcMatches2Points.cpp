#include "parameters.h"

void gcMatches2Points(vector<DMatch> matches, vector<Point2f>& _keypointsA_filtered, vector<Point2f>& _keypointsB_filtered,
				vector<KeyPoint> _keypointsA, vector<KeyPoint> _keypointsB)
{
	for( int i = 0; i < matches.size(); i++ )				
	{
		//-- Get the keypoints from the best matches
		_keypointsA_filtered.push_back( _keypointsA[ matches[i].queryIdx ].pt );
		_keypointsB_filtered.push_back( _keypointsB[ matches[i].trainIdx ].pt );						
	}
}