/* 
 * File:   VideoMiner.cpp
 * Author: joseph
 * 
 * Created on May 26, 2016, 8:03 AM
 */

#include "VideoMiner.h"
#include "cvutilities.h"
#include <sys/stat.h>
/*
 Source: Directory name of the URL list
 DEST: Directory path of the desired output
 */
VideoMiner::VideoMiner(std::string List, std::string Dest) {
    
    SOURCE_DIR = List;    
    DEST_DIR = Dest;
    
    //Setup auxillary parameters
    NEG_WAIT_SEC = 10;
    pause = false;
    read = false;    
    skip = false;
    vid_num = 0;
    frame = 0;
    last_time = 0;
    negative_count = 0;
}

VideoMiner::VideoMiner(const VideoMiner& orig) {
}

VideoMiner::~VideoMiner() {    
}

bool VideoMiner::init()
{        
    speed = "?";
    bool SUCCESS;
    PROGRESS_FILE = SOURCE_DIR + "Progress.txt";
    //Load previous progress
    SUCCESS = loadProgress(PROGRESS_FILE, vid_num, frame);
    if(!SUCCESS)        
        return false;    
    SUCCESS = detector.init();
    if(!SUCCESS)
        return false;
     
    detector.setDistanceMatrix(SOURCE_DIR+"Distance");
    /* DEBUG */
    detector.initCascade("../../Classifier/cascade.xml");    
    return true;
}

//Setup the positive and negative directories
bool VideoMiner::setupDir()
{    
    std::string dir_name = "Vid"+to_string(vid_num)+"/";
    IMG_WRITE_DIR = DEST_DIR + dir_name;
    int dir_err = mkdir((IMG_WRITE_DIR).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == dir_err)
    {
        //cout<<DEST_DIR + dir_name<<" Could not be created: exiting main - NOW\n";
        //return -1;
    }
    
    return true;
}

bool VideoMiner::updateVideo()
{    
    //Get the desired video URL
    if(!getNextVid(SOURCE_DIR+"URL_list.txt",URL,vid_num))    
        return false;
    
    //Set the current video name
    VID_NAME = "Vid"+to_string(vid_num)+".mp4";
    
    //Set the download command
    DLcommand = "youtube-dl -o \""+DEST_DIR+"Vid"+to_string(vid_num)+".mp4\" "+URL;
    
    //Download video
    system(DLcommand.c_str());
    
    // Set progress file to first frame
    updateProgress(PROGRESS_FILE, vid_num, 0);
    
    return true;
}

bool VideoMiner::scanVideo()
{
    capture.open(SOURCE_DIR + VID_NAME);
    //capture.open("/home/joseph/Desktop/pi_car_proj/video/Test1.mp4");
    if(!capture.isOpened())
        return false;
    
    //RESET variables
    negative_count = 0;
    last_time = 0;
    int spinnerIdx = 0;
    bool save = false;
    //main video loop    
    capture.set(CV_CAP_PROP_POS_FRAMES, frame);    //frame
    cv::Mat src,img, grey;
    std::vector<cv::Rect> Signs;
    long int img_counter = 0;
    cv::Rect FoundObjLoc;
    cv::Rect CropRect = cv::Rect(320, 150, 320, 480-150);
    double FPS_avg = 0;
    while(1)
    {
        std::chrono::high_resolution_clock::time_point current_time = std::chrono::high_resolution_clock::now();
        if(!capture.read(src))
        {
            std::cout<<"Video "<<vid_num<<" finished\n";
            break;
        }
        // Resize, convert to gray-scale and crop
        resize(src,img,cv::Size(640,480));        
        cvtColor(img, grey, CV_BGR2GRAY);        
        cv::Mat raw_croped(img,CropRect);   
        cv::Mat candidate;

        //feed img to frame scan
        int results = detector.ScanImage(raw_croped,candidate);
        
        std::string OutName;
        if(results == SIGN_FOUND)
        {
            // Positives will have a P in the filename to reduce manual sorting            
            OutName = to_string(vid_num) + "_" + 
                      to_string(frame) + "_" + 
                      "P" + detector.getSpeed()+".jpg";
            detector.getFound(FoundObjLoc);
            
            // Save larger resolution if the the source image is larger
            cv::Rect larger = detector.MapRectToParentScale(src,img,
                              FoundObjLoc,CropRect);
            cv::imwrite(IMG_WRITE_DIR + OutName,src(larger));            

        }        
        else if (results == SIGN_CLASSIFIED)
        {
            OutName = to_string(vid_num) + "_" + 
                      to_string(frame) + "_" + 
                      "N" + ".jpg";
            detector.getFound(FoundObjLoc);
            
            // Save larger resolution if the the source image is larger            
            cv::Rect larger = detector.MapRectToParentScale(src,img,
                              FoundObjLoc,CropRect);
            cv::imwrite(IMG_WRITE_DIR + OutName,src(larger));
            
        }        
        //cv::namedWindow("Video Feed",CV_WINDOW_KEEPRATIO);
        imshow("Video Feed",raw_croped);
        char key = cv::waitKey(1);
        // User resize window
        if(key == 'r')
            cvUtils::UsrCrop(img,CropRect);        
                               
        // Setup increment fps average
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - current_time);
        double FPS = 1.0/time_span.count();
        FPS_avg += FPS;
        int mult = 20;
        //Update video progress
        if(frame % mult*((int) capture.get(cv::CAP_PROP_FPS)) == 0)
        {   
            FPS_avg = FPS_avg/mult;
            updateProgress(PROGRESS_FILE, vid_num, frame);
            double ETA_S = (capture.get(cv::CAP_PROP_FRAME_COUNT) - frame) / FPS_avg;
            double perc = ((double) frame) / ((double) capture.get(cv::CAP_PROP_FRAME_COUNT));            
            printProgBar(perc*100.0,spinnerIdx, FPS_avg, ETA_S);
            FPS_avg = 0;
        }

        frame++;        
    }    
    //Video scan complete - update video number to get next video
    vid_num++;    
    frame = 0;
    img_counter = 0;
    //clean the video
    std::string clean_file_cmd = "rm " +SOURCE_DIR + VID_NAME;
    std::cout<<"clean using: "<<clean_file_cmd<<std::endl;
    system(clean_file_cmd.c_str());
    
    return true;
}

