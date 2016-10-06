# V-STaR
Visual-based Speed sign Targeting and Recognition

![alt tag](https://github.com/jchrismer/V-STaR/blob/master/GitHubDemoImg.jpg)

## Overview
V-STaR is a C++ program which automatically detects and labels US speed limit signs from video. 

## Motivation
V-STaR was created to aid in driver awareness by using a raspberry pi 3 to automatically recognizing and cataloguing speed limit signs, preventing the driver from missing or forgeting local speeds. Currently it's focus is on autonomous mining of speed limit signs images to be used for further refinement.

## How does it work?
- Videos are downloaded from a user curated document labeled "URL_list.txt"
- "URL_list.txt" is just a list of youtube URLs which are preselected by the user for mining/evaluation
- Progress is logged in "Progress.txt" which allows users to end the current session and resume on the last frame
- OpenCV loads images from a camera or video file
- Rectangles of similar size to a speed sign are sough usting a Fast Symmetric Radial Transform (FRST)
- Detected rectangles are sent to a trained LBP classifier to quickly determine if the object is a Speed sign
- Any object recognized by the classifier is sent to a Convolutional Neural Network (CNN) for detection
  - The CNN has 11 outputs corresponding to 25,30, ... 75
  - If the CNN's output exceeds a minimum convidence value the object is considered a speed limit sign and it's value is drawn on the screen
- V-STaR also has a Optical Character Recognition (OCR) based detection mode
  - This mode attempts to read the sign by segmenting text out from the classified object and passing it to a tesseract OCR
  - A minimum edit distance is used to determine how close the recognized text is to "SPEED" and "LIMIT"
  - Text that is within a minium edit distance is considered to be that of a sign it's value drawn on the screen
  - Although the OCR is significantly less accurate than the CNN, when combined with FRST, OCR mode can read any type of rectangular road sign, allowing for autonomous imaging databases to be built for futher trainig and refinement.
  
## Requirements
- Only tested on Linux (sys commands may not work on other operating systems)
- OpenCV
- Keras
- Theano
- Tesseract
- Python = 2.7

## Usage
./vid_miner [destination]

## Installation
- mkdir build
- cd build
- cmake ..
- make

## Want to learn more?
This repo is part of a project which can be found at: http://aerialarithmetic.blogspot.com/. Look under any post with "Raspberry Pi Speed Sign Detector" in the title to learn more about theory driving the project, future plans, issues and areas for improvments.
