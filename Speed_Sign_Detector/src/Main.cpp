#include "VideoMiner.h"
using namespace std;
int main(int argc, char** argv) {
    if(argc != 2)
    {
	cout<<"Invalid input arguments"<<endl;
	cout<<"Destination for video footage not specified, Example usage:"<<endl;
	cout<<"./vid_miner /home/pi/Desktop/Video/"<<endl;

	return -1;

    }    

    VideoMiner Miner("../../video/",argv[1]);	// Location of URL list, destination
    if(!Miner.init())
    {
        cout<<"Video miner NOT Initialized"<<endl;
        return -1;
    }        
    
    while(Miner.updateVideo())
    {
        Miner.setupDir();
        Miner.scanVideo();    
    }
}

