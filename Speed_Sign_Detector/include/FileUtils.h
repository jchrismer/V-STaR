/* 
 * File:   FIleUtils.h
 * Author: joseph
 *
 * Created on May 22, 2016, 3:35 PM
 */

#ifndef FILEUTILS_H
#define	FILEUTILS_H

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <string.h>

#define MAX_STRING 128

using namespace std;    //remove

#include <string>
#include <sstream>

inline std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


inline std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

inline bool loadProgress(std::string progressPath, unsigned int &video_num, long int &frame) {
    
    std::ifstream progress;
    progress.open(progressPath, ios_base::in);
    if (!progress.is_open()) {
        std::cout << "Progress txt file could not be opened at: " << progress << std::endl;
        std::cout << "Check given name or make sure progress file exists\n";
        return false;
    }

    std::string result = "";

    //Set file position to the end
    progress.seekg(0, std::ios_base::end);
    char ch = ' ';
    while (ch != '\n') {
        progress.seekg(-2, std::ios_base::cur);                        
        if ((int) progress.tellg() <= 0) {
            progress.seekg(0); 
            break;
        }
        progress.get(ch);
    }

    std::getline(progress, result);
    std::vector<std::string> last_line = split(result,',');
    //No previous progress found
    if(last_line.size() != 2)
    {
        video_num = 0;
        frame = 0;        
    }
    //Progress exists, load it into output
    else
    {
        video_num = stoi(last_line.at(0));
        frame = stol(last_line.at(1));        
    }    
        
    return true;
}

/*
 Finds the latest unprocessed video URL or file to pass to the main mining method
 * Returns 3 exit states as integers:
 * VIDEO_NEW = Unseen video so it must be downloaded and processed
 * VIDEO_PARTLY = Video has been partly processed but not finished (MUST BE ON FILE)
 * VIDEO_FILE_ERROR = Error in File format or opening List.txt
 */
inline bool getNextVid(string URL_LIST,string &URL,int vid_num)
{
    ifstream in;    //close called in destructor

    in.open(URL_LIST);    //fix
    if(!in.is_open())
    {
        std::cout<<"URL list file: "<<" Could not be opened\n";
        return false;        
    }
    char currLine[MAX_STRING];    
         
    //Get a line and process it's contents
    int line_num = 0;
    while(in.getline(currLine,MAX_STRING,'\n'))
    {
        string tmp(currLine);
        vector<string> Line = split(tmp,',');
        
        //URL only setup outputs and return
        if(Line.size() == 1)
        {
            if(vid_num == line_num)
            {
                URL = Line.at(0);                            
                return true;
            }
            line_num++;
        }

        else
        {
            std::cout<<"URL number: "<<vid_num<<" -- Could not be found in URL list file\n";
            return false;            
        }
    }
}

inline bool updateProgress(string PROGRESS_PATH, int video_name, int frame)
{
    ofstream outfile;    
    outfile.open (PROGRESS_PATH, ios::out | ios::app);     //fix
    
    if(!outfile.is_open())        
        return false;
    //Saves it as "[# as appeared in list file],frame"
    string OUT_LINE =to_string(video_name) +","+to_string(frame);
    outfile << OUT_LINE<<endl;    
    return true;
}

inline bool writeStringleOut(string PROGRESS_PATH, string out)
{
    ofstream outfile;
    outfile.open (PROGRESS_PATH, ios::out | ios::app);     //fix
    
    if(!outfile.is_open())
        return false;
    //Saves it as "[# as appeared in list file],frame"    
    outfile << out <<endl;
    return true;
}
inline void printProgBar( float fpercent,int &spinnerIdx, int fps,int ETA_S ){
  // Prepare ETA
    int hour = ETA_S / 3600;
    int minute = (ETA_S % 3600) / 60;
    int seconds = (ETA_S % 3600) % 60;
  
  std::string bar;
  int percent = fpercent;
  char spinner[4] = {'|','/','-','\\'};
  for(int i = 0; i < 50; i++){
    if( i < (percent/2)){
      bar.replace(i,1,"=");
    }else if( i == (percent/2)){
      bar.replace(i,1,">");
    }else{
      bar.replace(i,1," ");
    }
  }
  
  std::cout<< "\r" "Working "<<"("<<spinner[spinnerIdx]<<") ... " 
        << "[" << bar << "] ";
  std::cout.width( 3 );
  std::cout.precision(4);
  std::cout<< fpercent << "%  @"<<fps<<"fps "<<" ETA = "<<hour<<"h : "<<minute<<"min : "<<seconds<<"s  "<< std::flush;
  spinnerIdx = (spinnerIdx + 1) % 4;
}

#endif	/* FILEUTILS_H */

