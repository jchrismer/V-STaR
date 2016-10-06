/* 
 * File:   VideoMiner.h
 * Author: joseph
 *
 * Created on May 26, 2016, 8:03 AM
 */

#ifndef VIDEOMINER_H
#define	VIDEOMINER_H
#include "Sign_Detector.h"
#include "FileUtils.h"
#include <chrono>
#include <ctime>
#include <ratio>
//RULE: ALL DIRECTORIES (variables with DIR at the end) END WITH "/"

class VideoMiner {
public:
    VideoMiner(std::string List, std::string Dest);
    VideoMiner(const VideoMiner& orig);
    virtual ~VideoMiner();    
    bool init();
    bool setupDir();
    bool updateVideo();    
    bool scanVideo();
    long getFrame(){return frame;}
    unsigned int getVidNum(){return vid_num;}
    void writeObject(cv::Mat &obj);
private:    
    void cleanVideo();    
    std::string SOURCE_DIR,PROGRESS_FILE, DEST_DIR, DLcommand, URL,
            IMG_WRITE_DIR,NEG_DIR,VID_NAME;    
    unsigned int vid_num,negative_count;    
    long frame;    
    int NEG_WAIT_SEC;    
    bool pause, read, skip;        
    string speed;    
    Sign_Detector detector;
    cv::VideoCapture capture;
    unsigned long int last_time;
    std::chrono::high_resolution_clock::time_point start;
};

#endif	/* VIDEOMINER_H */

