#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

class CCamCalib
{
public:
	CCamCalib(int board_w = 9, int board_h = 6, int n_boards = 2, float cell_w = 0.035f, float cell_h = 0.035f);
	virtual ~CCamCalib();

	void LoadCalibParams (CvSize &image_size);
	bool FindChessboard(IplImage *src, IplImage *dst);
	void Undistort(IplImage *src, IplImage *dst);
	void CalibrateCamera(CvSize &image_size);
	
	CvMat* _image_points;
	CvMat* _object_points;
	CvMat* _point_counts;

	CvMat* _intrinsic_matrix;
	CvMat* _distortion_coeffs;

	IplImage* _mapx;
	IplImage* _mapy; 

	float _cell_w;	// 체스판에서 한 격자의 가로방향 넓이
	float _cell_h;	// 체스판에서 한 격자의 세로방향 넓이

	int _n_boards;	// 인식할 체스판 수를 지정한다.
	int _board_w;	// 체스판의 가로방향 코너 수
	int _board_h;	// 체스판의 세로방향 코너 수
	int _board_n;	// 가로 x 세로 방향의 코너 수
	int _board_total;
	int _successes;
};