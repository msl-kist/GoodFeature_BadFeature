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

	float _cell_w;	// ü���ǿ��� �� ������ ���ι��� ����
	float _cell_h;	// ü���ǿ��� �� ������ ���ι��� ����

	int _n_boards;	// �ν��� ü���� ���� �����Ѵ�.
	int _board_w;	// ü������ ���ι��� �ڳ� ��
	int _board_h;	// ü������ ���ι��� �ڳ� ��
	int _board_n;	// ���� x ���� ������ �ڳ� ��
	int _board_total;
	int _successes;
};