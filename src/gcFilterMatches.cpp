#include "parameters.h"


void gcFilterMatches(vector<DMatch> src, vector<DMatch> src_opp, vector<DMatch>& dst, 
					vector<KeyPoint> _keypointsA, vector<KeyPoint> _keypointsB)
{
	//std::vector<cv::Point2f> keypointsA_all, keypointsB_all; //while문 안에 없으면 tracking이 굉장히 느리게 반응함
	//std::vector<cv::Point2f> keypointsA_filtered, keypointsB_filtered; //while문 안에 없으면 tracking이 굉장히 느리게 반응함
	//vector<Point2f> mpts_1, mpts_2;		// train이 1 query가 2
	//CvPoint dst_corners[4];
	
#if Matching_Method_6
	//imgA ---> imgB search
	std::vector<DMatch> good_matches;
	for( int i = 0; i < src.size(); i++ )
	{
		if(   ( src[i].distance <= distance_threshold )   ) good_matches.push_back( src[i] );
	}			
	if (good_matches.size() == 0)
	{
		//std::cout<< " --(!) No good matches found " << std::endl; 
		//continue; 
	}		
	std::vector<DMatch> better_matches;
	float ratio = 0;
	for(int j=0; j<_keypointsB.size(); j++)				
	{						
		std::vector<DMatch> matches_overlap;
		for(int i=0; i<good_matches.size(); i++)	
		{								
			if(good_matches[i].trainIdx == j) 
			{					
				matches_overlap.push_back( good_matches[i] );
			}
		}
		if(matches_overlap.size() >= 2)
		{
			sort(matches_overlap.begin(),matches_overlap.end());
			ratio = matches_overlap[0].distance / matches_overlap[1].distance;
			if( ratio  < ratio_threshold)
			{
				better_matches.push_back(matches_overlap[0]);
			}								
		}
		else if(matches_overlap.size()==1)
			better_matches.push_back(matches_overlap[0]);
	}		

	//imgB ---> imgA search
	std::vector<DMatch> good_matches_B;		
	for( int i = 0; i < src_opp.size(); i++ )
	{
		if(   ( src_opp[i].distance <= distance_threshold )   ) good_matches_B.push_back( src_opp[i] );
	}			
	if (good_matches_B.size() == 0)
	{
		//std::cout<< " --(!) No good matches found " << std::endl; 
		//continue; 
	}	
	std::vector<DMatch> better_matches_B;		
	for(int j=0; j<_keypointsA.size(); j++)				
	{						
		std::vector<DMatch> matches_overlap_B;
		for(int i=0; i<good_matches_B.size(); i++)	
		{								
			if(good_matches_B[i].trainIdx == j) 
			{					
				matches_overlap_B.push_back( good_matches_B[i] );
			}
		}
		if(matches_overlap_B.size() >= 2)
		{
			sort(matches_overlap_B.begin(),matches_overlap_B.end());
			ratio = matches_overlap_B[0].distance / matches_overlap_B[1].distance;
			if(( ratio ) < ratio_threshold)
			{
				better_matches_B.push_back(matches_overlap_B[0]);
			}								
		}
		else if(matches_overlap_B.size()==1)
			better_matches_B.push_back(matches_overlap_B[0]);
	}		

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		

	//std::vector<DMatch> best_matches;
	for(int j=0 ; j<better_matches_B.size() ; j++)
	{		
		for(int i=0 ; i<better_matches.size() ; i++)
		{				
			if(  (better_matches[i].queryIdx == better_matches_B[j].trainIdx) && (better_matches[i].trainIdx == better_matches_B[j].queryIdx)  )
				dst.push_back(better_matches[i]);				
		}										
	}
#elif Matching_Method_4

	for(int j=0 ; j<src_opp.size() ; j++)
	{		
		for(int i=0 ; i<src.size() ; i++)
		{				
			if(  (src[i].queryIdx == src_opp[j].trainIdx) && (src[i].trainIdx == src_opp[j].queryIdx)  )
				dst.push_back(src[i]);				
		}										
	}

#elif Matching_Method_5
	std::vector<DMatch> good_matches;
	for( int i = 0; i < src.size(); i++ )
	{
		if(   ( src[i].distance <= distance_threshold )   ) good_matches.push_back( src[i] );
	}
	if (good_matches.size() == 0)
	{
		std::cout<< " --(!) No good matches found " << std::endl; 
		//continue; 
	}		

	std::vector<DMatch> good_matches_B;		
	for( int i = 0; i < src_opp.size(); i++ )
	{
		if(   ( src_opp[i].distance <= distance_threshold )   ) good_matches_B.push_back( src_opp[i] );
	}			
	if (good_matches_B.size() == 0)
	{
		std::cout<< " --(!) No good matches found " << std::endl; 
		//continue; 
	}	

	for(int j=0 ; j<good_matches_B.size() ; j++)
	{		
		for(int i=0 ; i<good_matches.size() ; i++)
		{				
			if(  (good_matches[i].queryIdx == good_matches_B[j].trainIdx) && (good_matches[i].trainIdx == good_matches_B[j].queryIdx)  )
				dst.push_back(good_matches[i]);				
		}										
	}
#endif

}