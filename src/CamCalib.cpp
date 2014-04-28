#include "CamCalib.h"


CCamCalib::CCamCalib (int board_w, int board_h, int n_boards, float cell_w, float cell_h)
	: _board_w(board_w), _board_h(board_h), _n_boards(n_boards), _cell_w(cell_w), _cell_h(cell_h)
{
	_board_n = _board_w*_board_h;

	// ü�������κ��� ã�� �ڳʸ� ������ ������� �Ҵ�
	_image_points  = cvCreateMat(_n_boards*_board_n, 2, CV_32FC1);
	_object_points = cvCreateMat(_n_boards*_board_n, 3, CV_32FC1);
	_point_counts  = cvCreateMat(_n_boards, 1, CV_32SC1);

	//Intrinsic Matrix - 3x3			   Lens Distorstion Matrix - 4x1
	//	[fx 0 cx]							[k1 k2 p1 p2   k3(optional)]
	//	[0 fy cy]
	//	[0  0  1]

	_intrinsic_matrix  = NULL;
	_distortion_coeffs = NULL;

	_mapx = NULL;
	_mapy = NULL;

	_successes = 0;
}

CCamCalib::~CCamCalib(void)
{
	cvReleaseMat(&_object_points);
	cvReleaseMat(&_image_points);
	cvReleaseMat(&_point_counts);
	
	if (_intrinsic_matrix)  cvReleaseMat(&_intrinsic_matrix);
	if (_distortion_coeffs) cvReleaseMat(&_distortion_coeffs);

	if (_mapx) cvReleaseImage(&_mapx);
	if (_mapy) cvReleaseImage(&_mapy);
}

void CCamCalib::LoadCalibParams (CvSize &image_size)
{
	// ���Ϸ� ����� ������İ� �ְ� ����� �ҷ�����
	_intrinsic_matrix  = (CvMat *)cvLoad("data\\cam\\Intrinsics.xml");
	_distortion_coeffs = (CvMat *)cvLoad("data\\cam\\Distortion.xml");

	std::cout<<"rrrrr"<<std::endl;
	if (_intrinsic_matrix && _distortion_coeffs) {
		std::cout<<"rrrrr"<<std::endl;
		// �ְ� ���Ÿ� ���� ������ ����
		_mapx = cvCreateImage( image_size, IPL_DEPTH_32F, 1 );
		_mapy = cvCreateImage( image_size, IPL_DEPTH_32F, 1 );

		// �ְ� ���Ÿ� ���� ������ ����
		cvInitUndistortMap (_intrinsic_matrix, _distortion_coeffs, _mapx, _mapy);

		_successes = _n_boards + 1;
	}
}

bool CCamCalib::FindChessboard(IplImage *src, IplImage *dst)
{
	IplImage *gray = cvCreateImage (cvGetSize(src), IPL_DEPTH_8U, 1);
	
	cvCvtColor(src, gray, CV_BGR2GRAY);

	// ü���� �ڳ� ã��
	CvPoint2D32f* corners = new CvPoint2D32f[_board_n];
	int corner_count = 0;
	int found = cvFindChessboardCorners(src, cvSize(_board_w, _board_h), corners, &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );

	// ����� �ڳʷκ��� �����ȼ� ��Ȯ���� �ڳ� ��ǥ�� ���Ѵ�.
	cvFindCornerSubPix (gray, corners, corner_count, cvSize(11,11), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

	// �ڳʸ� dst �̹����� �׸���.
	cvDrawChessboardCorners (dst, cvSize(_board_w, _board_h), corners, corner_count, found);

	// �ڳʸ� ���������� ã�Ҵٸ�, �ڳ� �����͸� �����Ѵ�.
	bool ret = false;
	if (found && corner_count == _board_n) {
		for( int i=_successes*_board_n, j=0; j<_board_n; ++i, ++j ) {
			CV_MAT_ELEM(*_image_points, float, i, 0) = corners[j].x;
			CV_MAT_ELEM(*_image_points, float, i, 1) = corners[j].y;
			CV_MAT_ELEM(*_object_points,float, i, 0) = (float)(j%_board_w)*_cell_w;
			CV_MAT_ELEM(*_object_points,float, i, 1) = (float)(_board_h - j/_board_w - 1)*_cell_h;
			CV_MAT_ELEM(*_object_points,float, i, 2) = 0.0f;
		}
		CV_MAT_ELEM(*_point_counts, int, _successes, 0) = _board_n; 
		
		ret = true;
	}	

	delete [] corners;
	cvReleaseImage(&gray);  
	return ret;
}

void CCamCalib::CalibrateCamera(CvSize &image_size)
{
	if (_intrinsic_matrix)  cvReleaseMat(&_intrinsic_matrix);
	if (_distortion_coeffs) cvReleaseMat(&_distortion_coeffs);

	if (_mapx) cvReleaseImage(&_mapx);
	if (_mapy) cvReleaseImage(&_mapy);

	_intrinsic_matrix  = cvCreateMat(3, 3, CV_32FC1);
	_distortion_coeffs = cvCreateMat(4, 1, CV_32FC1);

	// ���� �Ÿ� ������ 1.0���� �����Ͽ� ��������� �ʱ�ȭ
	CV_MAT_ELEM( *_intrinsic_matrix, float, 0, 0 ) = 1.0f;
	CV_MAT_ELEM( *_intrinsic_matrix, float, 1, 1 ) = 1.0f;

	// ���� ī�޶� �����Լ�
	cvCalibrateCamera2 (_object_points, _image_points, _point_counts, image_size, _intrinsic_matrix, _distortion_coeffs, NULL, NULL, 0);

	// ���� ��İ� �ְ� ����� ���Ϸ� ����
	cvSave("Intrinsics.xml", _intrinsic_matrix);
	cvSave("Distortion.xml", _distortion_coeffs);

	// �ְ� ���Ÿ� ���� ������ ����
	_mapx = cvCreateImage( image_size, IPL_DEPTH_32F, 1 );
	_mapy = cvCreateImage( image_size, IPL_DEPTH_32F, 1 );

	// �ְ� ���Ÿ� ���� ������ ����
	cvInitUndistortMap (_intrinsic_matrix, _distortion_coeffs, _mapx, _mapy);
}

void CCamCalib::Undistort(IplImage *src, IplImage *dst)
{
	assert (_mapx);
	assert (_mapy);

	// ī�޶� �Է¿���(src)���� �ְ��� ������ ����(dst)�� �����.
	cvRemap(src, dst, _mapx, _mapy );			// Undistort image
}
