#include "parameters.h"

void gcHomography(bool computeHomography, Mat src, Mat &dst, 
				  vector<Point2f> keypointsA_filtered, vector<Point2f> keypointsB_filtered, Scalar color)
{
	if (computeHomography && (keypointsA_filtered.size() > 5) && (keypointsB_filtered.size() > 5) )
	{
		//Mat H = findHomography(Mat(mpts_1), Mat(mpts_2), RANSAC, 2);
		Mat H = findHomography(keypointsA_filtered, keypointsB_filtered, RANSAC, 2);

		//Training Image 박스 그리기
		//-- Get the corners from the image_1 ( the object to be "detected" )
		vector<Point2f> obj_corners1(4);
		obj_corners1[0] = cvPoint(0,0); 
		obj_corners1[1] = cvPoint( src.cols, 0 );
		obj_corners1[2] = cvPoint( src.cols, src.rows ); 
		obj_corners1[3] = cvPoint( 0, src.rows );	

		//Convert Object Corners to Transformed Object Corners Using Homography Matrix Information
		vector<Point2f>				dst_matching_corners(4);
		perspectiveTransform( obj_corners1, dst_matching_corners, H);

		line( dst, dst_matching_corners[0] + Point2f( src.cols, 0), dst_matching_corners[1] + Point2f( src.cols, 0), color, 4 );
		line( dst, dst_matching_corners[1] + Point2f( src.cols, 0), dst_matching_corners[2] + Point2f( src.cols, 0), color, 4 );
		line( dst, dst_matching_corners[2] + Point2f( src.cols, 0), dst_matching_corners[3] + Point2f( src.cols, 0), color, 4 );
		line( dst, dst_matching_corners[3] + Point2f( src.cols, 0), dst_matching_corners[0] + Point2f( src.cols, 0), color, 4 );			
	}
}