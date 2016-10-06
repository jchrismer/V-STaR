#ifndef CVUTILITIES_H
#define CVUTILITIES_H

#include <math.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <assert.h>
#include <iostream>

typedef cv::Mat Mat;

namespace cvUtils {            
	float round(float d);
	bool inImage(cv::Point point, cv::Size imageSize);
	bool inRanges(float value, std::vector<cv::Range> ranges);
	void saveOverlay(const char* filename, Mat img1, Mat img2, float blending = 0.5);
	void drawRegPolygon(Mat image, int numSides, float radius, cv::Point center, float rotation, cv::Scalar color = cv::Scalar(0, 255, 0), int thickness = 1);
    void hueAdd(Mat image, int value, Mat mask, int upperBound = 180);
	void warpPerspective(const Mat& src, Mat& dst, const Mat& M0, cv::Size dsize, int flags = cv::INTER_LINEAR, int borderType = cv::BORDER_CONSTANT, const cv::Scalar& borderValue = cv::Scalar(), cv::Point origin = cv::Point(0,0));
	Mat resizeCanvas(Mat input, cv::Scalar fill, int left = 0, int right = 0, int up = 0, int down = 0);
	Mat resizeCanvas(Mat input, cv::Scalar fill, int* parameters);
	Mat autoCrop(Mat input, int *parameters = NULL);
	Mat blendMask(Mat foreground, Mat background, Mat mask);
        cv::Rect constrainRectToImage(cv::Mat &Img, cv::Rect &r);
        
        // user video cropping
        extern cv::Rect UsrRect;
        extern bool bDraw;
        extern cv::Rect r;
        extern cv::Point base;

        extern Mat img;
        extern Mat layer;
        extern Mat working;
        extern Mat CropedImg;
        
        void CallBackFunc(int event, int x, int y, int flags, void* userdata);
        void UsrCrop(Mat &in,cv::Rect &usrSel);        
}

#endif // CVUTILITIES_H
