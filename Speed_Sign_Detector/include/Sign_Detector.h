/* 
 * File:   Sign_Detector.h
 * Author: joseph
 *
 * Created on May 4, 2016, 11:16 AM
 */

#ifndef SIGN_DETECTOR_H
#define	SIGN_DETECTOR_H
#include "PythonEmbeded.h"
#include <algorithm>    // std::iter_swap
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <tesseract/baseapi.h>
#include <opencv2/core/types_c.h>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <sstream>
#include <chrono>
#include "FileUtils.h"
#include "foundshape.h"
#include "cvutilities.h"
#include "FRST.h"

#define MIN_CONF 50.0
#define IMG_WIDTH 240
#define IMG_HEIGHT 190
#define NO_SIGN_FOUND 1
#define SIGN_FOUND 2
#define SIGN_CLASSIFIED 3
#define MODE_OCR 10
#define MODE_CNN 11
const std::string DEMO_DIR = "/home/joseph/Blog/SpeedSignWriteUp/Images/";
struct ROI {        
        ROI(cv::Rect in_rect, Mat in_Mat) {
            rect = in_rect;
            Regions = in_Mat;
            Line_Num = 0;
        }

        cv::Rect rect;
        Mat Regions;
        int Line_Num;
        bool isSpeedNumber;
};

class Sign_Detector
{

public:
    Sign_Detector();
    ~Sign_Detector();
    int ScanImage(cv::Mat &scene, cv::Mat &candidate);    
    bool init();
    std::string getSpeed(){return speed;}
    void setDistanceMatrix(std::string filename);
    bool initCascade(std::string CascadeLocAndName);
    Mat extractROI(cv::Mat image,cv::Size aspectRatio,foundShape bestShapes,cv::Rect &BB);
    void ScanForText(std::vector<ROI> &outputRegions,cv::Mat &img_in, bool show);
    void runCascade(cv::Mat image,std::vector<cv::Rect> &results);
    cv::Rect extractBBFromParent(cv::Rect Parent, cv::Rect child,cv::Point old_size);
    int constrain(int in, int lowerBoundary, int upperBoundary);
    cv::Rect doubleBB(Mat img,cv::Rect r);
    void setMode(unsigned short int usrMode);
    void getFound(cv::Rect &input){input = bb;}
    cv::Rect MapRectToParentScale(cv::Mat &ParentImg, cv::Mat &ChildImg, cv::Rect &BB, cv::Rect &cropping);
    int RunCNN(cv::Mat img);
    int RunOCR(cv::Mat img);
    bool OCR_found_Sign;
    bool CNN_found_Sign;
    std::string OCR_speed;
private:
    std::string runOCR(tesseract::TessBaseAPI *tess,Mat image, bool &conf);    
    void lexInsSort(std::vector<ROI> &Sort);
    bool lexGT(ROI p1, ROI p2);    
    bool TranslateFoundTxt(std::vector<ROI> &TxtRegions, tesseract::TessBaseAPI *tess, 
                   int &loc,std::string &speed);
    float edit_distance(const std::string& s1, const std::string& s2, float Distance_Mat[36][36]);    
    void constrainRect(cv::Rect &rect,cv::Point boundary);       
    
    //Variables
    cv::Size aspectRatio;
    int minWidth; //30
    int maxWidth; //70
    int maxShapes; //3
    int voteThreshold; //10
    int widthStep; //20
    int angleTolerance; //5    
    FRST frst;
    tesseract::TessBaseAPI tesseract_api;
    std::vector<ROI> TxtRegions;
    cv::Rect bb;
    std::string speed;    
    float dis_mat[36][36];
    int frame;    
    bool cascade_init;
    cv::CascadeClassifier sign_cascade;
    std::vector<cv::Rect> CascadeResults;    
    unsigned int classifier_count;
    PythonEmbeded pythonModule;
    unsigned short int MODE;
    
};

#endif	/* SIGN_DETECTOR_H */

