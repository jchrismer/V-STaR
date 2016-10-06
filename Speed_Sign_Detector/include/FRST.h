#ifndef FRST_H
#define FRST_H

#include <math.h>
#include <vector>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cvutilities.h"
#include "foundshape.h"
#include <iostream>
#include <chrono>
#include <cstdlib>

typedef cv::Mat Mat;

class FRST
{
public:
	FRST();
	FRST(Mat image);
	void setImage(Mat image);
	std::vector<foundShape> findRegPolygons(int numSides, int minApothem, int maxApothem, int maxShapes = 10, int voteThreshold = 0, int apothemStep = 2, int baseRotation = 0, int angleTolerance = 5);
	std::vector<foundShape> findRectangles(cv::Size aspectRatio, int minWidth, int maxWidth, int maxShapes = 10, int voteThreshold = 0, int widthStep = 2, int angleTolerance = 5);
	bool getSeenBefore();
	void setSeenBefore(bool seenBefore);
        void extractROI(cv::Mat image, cv::Size aspectRatio, foundShape bestShapes, cv::Rect &BB);
                
private:

	Mat voteRegPoly(int numSides, int apothem, int rotation, int angleTolerance);
	Mat voteRectangle(cv::Size aspectRatio, int width, int angleTolerance);
	void findGradients(float* thresholdValue);
	void angleRanges(std::vector<cv::Range>* ranges, int numSides, int rotation, int angleTolerance);
	void findShapes(std::vector<foundShape>* bestShapes, std::vector<Mat> votes, std::vector<int> sizes, int maxShapes = 10, int voteThreshold = 0);
	void vote(Mat votes, cv::Vec2f unitGradient, cv::Point basePoint, int w, int distance);
	bool seenBefore(foundShape shape, int allowedShift = 5);
        //modified        
	Mat image;
	float currentThreshold;
	int callCounter;
	std::vector<foundShape> previousFindings;
	bool seenBeforeCheck;                        
        Mat xSobel, ySobel;
        Mat magnitudes;
        Mat vote_mat;        
};
#endif // SHAPEFINDER_H
