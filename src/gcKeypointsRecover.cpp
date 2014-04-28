#include "parameters.h"

void gcKeypointsRecover(vector<KeyPoint> src, vector<KeyPoint> &dst, vector<KeyPoint> keypoints_original)
{
	for(int j=0 ; j<keypoints_original.size() ; j++)	
	{
		bool check = false;
		for(int i=0 ; i<src.size() ; i++)	
		{			
			if(  (keypoints_original[j].pt.x == src[i].pt.x)  &&  (keypoints_original[j].pt.y == src[i].pt.y)   )
			{
				dst.push_back(src[i]);
				check = true;
			}					
		}
		if(!check)
		{
			dst.push_back(cv::KeyPoint(0, 0, 0, 0, 0, 0));
		}		
	}
}

