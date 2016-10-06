/******************************************************************************* 
 * File:   Sign_Detector.cpp
 * Author: Joseph
 * 
 * Class which scans a frame for a speed sign. First uses the Fast Symmetric 
 * Radial Transform (FSRT.h) class to hypothesize likely regions containing
 * speed limit signs. Then segments text blocks out of the regions. Signs are
 * then classified using Tesseract OCR.
 * 
 ******************************************************************************/
#include "Sign_Detector.h"
#include <opencv2/core/mat.hpp>

/*******************************************************************************
 * Sign_Detector constructor:
 * Initializes a Sign_Detector instance with preset variables for use in FSRT
 * and TLD.
 ******************************************************************************/
Sign_Detector::Sign_Detector() {
    
    // Init FRST parameters
    aspectRatio = cv::Size(6, 5);   // aspect ratio of rectangles to search for
    minWidth = 30;          // Minimum rectangle width to search for
    maxWidth = 30;          // Maximum rectangle width to search for
    maxShapes = 5;          // Total number of candidate signs FRST returns
    voteThreshold = 10;     // Minimum voting threshold to qualify a center
    widthStep = 5;          // width pixel step size (not used)
    angleTolerance = 5;     // Maximum angle by which a rectangle can be rotated

    cascade_init = false;
    classifier_count = 0;
}

Sign_Detector::~Sign_Detector() {
    pythonModule.CleanPython();
}

/*******************************************************************************
 * init:
 * Initializes a tesseract class member and sets appropriate letters for OCR to 
 * consider.
 * 
 * Returns: 
 * - True if tesseract was succesfully initialized
 * - False otherwise
 ******************************************************************************/
bool Sign_Detector::init() {
    int SUCCESS = tesseract_api.Init(NULL, "eng");
    tesseract_api.SetVariable("tessedit_char_whitelist",
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    if (SUCCESS != 0)
        return false;

    // Setup python interpreter (Used for neural networking)
    if (pythonModule.setupPython() == 0)
        return false;

    MODE = MODE_CNN;

    return true;
}


// Sets sign recognition mode. Currently 2: OCR and CNN
void Sign_Detector::setMode(unsigned short int usrMode) {
    if (usrMode != MODE_CNN || usrMode != MODE_OCR) {
        std::cout << "Error: user selected mode " << usrMode << " not supported\n";
        std::cout << "Current supported modes are: " << MODE_CNN << " = CNN, " << MODE_OCR << " = OCR" << std::endl;
        return;
    }

    MODE = usrMode;
}

/*******************************************************************************
 * ScanForText:
 * Scans through an input image for text blocks and stores them in a ROI vector.
 * Text blocks are rectangles which are hypothesized use contain man made road 
 * sign text.
 * inputs:
 * @outputRegions - vector<ROI> containing all found text blocks in given image
 * @img_in        - cv::Mat* representing the image to be searched for text
 * @show          - boolean for debugging. When true shows all found text blocks
 * 
 ******************************************************************************/
void Sign_Detector::ScanForText(std::vector<ROI> &outputRegions, cv::Mat &img_in, bool show) {
    outputRegions.clear();

    int scale = 3;
    int W = scale * img_in.size().width;
    int H = scale * img_in.size().height;

    cv::Size size(W, H); //the dst image size,e.g.100x100          
    cv::Mat large;
    cv::resize(img_in, large, size);
    //cv::imshow("FSRT",large);
    //cv::waitKey(0);
    Mat temp; //debuging
    large.copyTo(temp);

    // morphological gradient (difference between dilation and erosion)
    cv::Mat grad;
    cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(large, grad, cv::MORPH_GRADIENT, morphKernel);

    //cv::imshow("Morphological Gradient",grad);
    //cv::waitKey(0);
    //cv::imwrite(DEMO_DIR+"Morphological_Gradient"+".jpg",grad);

    // binarize
    cv::Mat bw;
    cv::threshold(grad, bw, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // connect horizontally oriented regions
    cv::Mat connected;
    morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1)); //3,1
    cv::morphologyEx(bw, connected, cv::MORPH_CLOSE, morphKernel);

    //cv::imshow("Extend Horizontal",connected);
    //cv::waitKey(0);
    //cv::imwrite(DEMO_DIR+"Extend_Horizontal"+".jpg",connected);
    // find contours
    cv::Mat mask = cv::Mat::zeros(bw.size(), CV_8UC1);
    std::vector<std::vector < cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    
    /*
    In the case of night driving the image could be black. In which case
    there are no components (heirarchy or contours have size 0). This case is
    handled below.
    */
    if (contours.size() == 0 || hierarchy.size() == 0)
        return;

    // filter contours
    for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
        cv::Rect rect = cv::boundingRect(contours[idx]);
        cv::Mat maskROI(mask, rect);
        maskROI = cv::Scalar(0, 0, 0);
        // fill the contour
        cv::drawContours(mask, contours, idx, cv::Scalar(255, 255, 255), CV_FILLED); //necessary?
        // ratio of non-zero pixels in the filled region        
        double r = (double) cv::countNonZero(maskROI) / (rect.width * rect.height);
        if (r > .3 // 30% of the suspected block must be filled with text
                &&
                (rect.height > 13 && rect.width > 15) 
                &&
                //rectangles which take up more than 40% are removed
                rect.area() / (W * H + 0.0) < 0.4
                ) {
            //Create a new ROI and push it into the region vector
            outputRegions.push_back(ROI(rect, cv::Mat(large, rect)));
        }
    }

    //Gets caught in infinite loop without
    if (outputRegions.size() == 0)
        return;

    // Distance threshold. If below rectangles will be merged
    int Ty = 20; //20px Height threshold
    int Tx = 7; //7px Horizontal threshold

    //Merge rectangles
    for (int i = 0; (i < outputRegions.size() - 1); i++) {
        for (int j = i + 1; j < outputRegions.size(); j++) {
            //check relative height
            if (abs(outputRegions.at(i).rect.y - outputRegions.at(j).rect.y) <= Ty) {
                //check for intersection
                if ((outputRegions.at(i).rect & outputRegions.at(j).rect).area() > 0) {
                    //rectangles intersect therefore merge
                    outputRegions.at(i).rect = outputRegions.at(i).rect | outputRegions.at(j).rect;
                    //Update regions
                    outputRegions.at(i).Regions = Mat(large, outputRegions.at(i).rect);

                    //remove element j
                    outputRegions.erase(outputRegions.begin() + j);
                    continue; //prevents accessing removed elements
                }//check if j is strictly contained in i, remove j if so
                else if ((outputRegions.at(i).rect & outputRegions.at(j).rect) == outputRegions.at(i).rect) {
                    outputRegions.erase(outputRegions.begin() + j);
                    continue;
                }

                //Find left most and right most rectangles
                cv::Rect *Left_most;
                cv::Rect *Right_most;
                int Right_most_idx;
                int Left_most_idx;
                if (outputRegions.at(i).rect.x < outputRegions.at(j).rect.x) {
                    Left_most = &outputRegions.at(i).rect;
                    Right_most = &outputRegions.at(j).rect;
                    Right_most_idx = j;
                    Left_most_idx = i;
                } else {
                    Left_most = &outputRegions.at(j).rect;
                    Right_most = &outputRegions.at(i).rect;
                    Right_most_idx = i;
                    Left_most_idx = j;
                }
                //Check distance between left most and right most rectangles
                int D = abs(Left_most->x + Left_most->width - Right_most->x);
                if (D <= Tx) {
                    //merge rectangles
                    Left_most->width += D + Right_most->width;

                    //constrain rectangle                        
                    constrainRect(outputRegions.at(Left_most_idx).rect
                            , cv::Point(large.size().width, large.size().height));

                    outputRegions.at(Left_most_idx).Regions =
                            cv::Mat(large, outputRegions.at(Left_most_idx).rect);
                    //remove right most rectangle from list                    
                    outputRegions.erase(outputRegions.begin() + Right_most_idx);
                    //Handles case where i was the right most rectangle
                    j = i;
                }
            }
        }
    }

    // lexicographically sort results
    lexInsSort(outputRegions);

    //DEBUGING
    for (int i = 0; i < outputRegions.size(); i++)
        if (show) {
            cv::rectangle(temp, outputRegions.at(i).rect, cv::Scalar(0, 255, 0), 2);
            imshow("Txt Regions", temp);
            cv::waitKey(0);
        }
    // Assign leveling
    int Line_Num = 0;
    for (int i = 1; i < outputRegions.size(); i++) {
        if (abs(outputRegions.at(i).rect.y - outputRegions.at(i - 1).rect.y) > Ty)
            Line_Num++;
        outputRegions.at(i).Line_Num = Line_Num;
    }
    //imshow("segmented",temp2);
    //imshow("merged", temp);
    //waitKey(0);
}

/*******************************************************************************
 * runCascade:
 * Runs the image through the cascade classifier ONLY.
 * 
 * inputs:
 * @image            - cv::Mat image input image
 * @BB               - cv::Rect resulting BB from classifier
 * 
 ******************************************************************************/
void Sign_Detector::runCascade(cv::Mat image, std::vector<cv::Rect> &results) {
    if (cascade_init)
        sign_cascade.detectMultiScale(image, results);
    //handle error
}

/*******************************************************************************
 * constrainRect:
 * Constrains a bounding box to be inside the given image boundaries. Prevents
 * assertion errors.
 * 
 * inputs:
 * @rect            - cv::Rect output containing the adjusted rectangle
 * @boundary        - cv::Point desired boundary to constrain the rectangle
 * 
 ******************************************************************************/
void Sign_Detector::constrainRect(cv::Rect &rect, cv::Point boundary) {
    if ((rect.width + rect.x) > boundary.x)
        rect.width = boundary.x - rect.x;
    if ((rect.height + rect.y) > boundary.y)
        rect.height = boundary.y - rect.y;
}

/*******************************************************************************
 * lexGT:
 * Lexicographical greater than for use in sorting. If p1 and p2 are 2d points,
 * then this function preforms the logical comparison:
 *  p1 >L p2
 * Where ">L" is "Lexicographical greater than"
 * This function is needed to sort the text blocks ScanForText uses. It allows 
 * for some locational context to be used in the OCR.
 * inputs:
 * @p1            - ROI representing the left hand side of the (Lex g.t) comparison
 * @p2            - ROI representing the right hand side of the (Lex g.t) comparison
 * 
 * Returns: bool
 * @True  - If p1 >L p2
 * @false - otherwise 
 ******************************************************************************/
bool Sign_Detector::lexGT(ROI p1, ROI p2) {
    int Ty = 20; //pixel height threshold
    //p1 is right of p2 or sufficiently beneath it (Screen Coordinates)
    if (p1.rect.x > p2.rect.x || abs(p1.rect.y - p2.rect.y) > Ty)
        return true;

    return false;
}

//Lexicographic insertion sort
void Sign_Detector::lexInsSort(std::vector<ROI> &Sort) {
    unsigned int len = Sort.size();
    for (int i = 0; i < len; i++) {
        int j = i;
        while (j > 0 && lexGT(Sort.at(j - 1), Sort.at(j))) {
            std::iter_swap(Sort.begin() + (j - 1), Sort.begin() + j);
            j--;
        }
    }
}

/*******************************************************************************
 * runOCR:
 * Runs the Optical Character Recognition method in tesseract on an input image.
 * Small images work best.
 * inputs:
 * @tess          - tesseract::TessBaseAPI Tesseract object
 * @image         - cv::Mat of the image to be scanned by tesseract
 * @Bconf         - bool Output of if the scanned image met the required confidence
 *                  score.
 * 
 * Returns: std::string
 * String of what the OCR found
 ******************************************************************************/
std::string Sign_Detector::runOCR(tesseract::TessBaseAPI *tess, cv::Mat image, bool &Bconf) {
    //Convert to BW
    Mat BW_thresh;
    Bconf = false;
    threshold(image, BW_thresh,
            0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    //imshow("Black and White", BW_thresh);
    //cv::waitKey(0);
    //imwrite(DEMO_DIR + "BW_Threshold_error.jpg",BW_thresh);

    tess->SetImage((uchar*) BW_thresh.data,
            BW_thresh.cols,
            BW_thresh.rows,
            1,
            BW_thresh.cols);

    //temp_ray is used to prevent a memory leak (see GetUTF8Text in baseapi.h)
    char * temp_ray = tess->GetUTF8Text();
    std::string out = std::string(temp_ray);
    delete [] temp_ray;

    /*  CONFIDENCE value  */
    tesseract::ResultIterator* ri = tess->GetIterator();
    if (ri != 0) {
        char* symbol = ri->GetUTF8Text(tesseract::RIL_WORD);

        if (symbol != 0) {
            float conf = ri->Confidence(tesseract::RIL_WORD);
            if (conf > MIN_CONF) {
                //std::cout << symbol << "\tconf: " << conf << "\n"; 
                Bconf = true;
            }

        }
        delete[] symbol;
    }
    //clean up
    delete ri;
    out.erase(std::remove(out.begin(), out.end(), '\n'), out.end());
    return out;
}

/*******************************************************************************
 * TranslateFoundTxt:
 * Runs through regions of interests found by the FRST to search for signs and 
 * their text values.
 * 
 * inputs:
 * @TxtRegions  - vector<ROI> regions of interest to scan through
 * @tess        - tesseract::TessBaseAPI* point of the tesseract object that will
 *                  be used to scan with (REDUDANT AND SHOULD BE REMOVED!)
 * @loc         - int output Location of the potential speed text in the regions 
 *                of interest
 * @speed       - std::string output of the speed limit text value
 * 
 * Returns: bool
 * @True - valid speed limit sign was detected
 * @False - No speed limit sign found
 ******************************************************************************/
bool Sign_Detector::TranslateFoundTxt(std::vector<ROI> &TxtRegions,
        tesseract::TessBaseAPI *tess,
        int& loc, std::string &speed) {
    int Last_Line_Num = 0;
    bool conf;
    bool SIGN_TEXT = false;
    int TXT_LOC = 0;
    /* Run each region of interest through the OCR and check its edit distance 
       to "SPEED" or "LIMIT" if it is close then flag the image as a sign and 
       attempt to read the speed.
    */
    for (int TxtRegionIdx = 0; TxtRegionIdx < TxtRegions.size(); TxtRegionIdx++) {
        std::string out = runOCR(tess, TxtRegions.at(TxtRegionIdx).Regions, conf);
        //std::cout<<out<<std::endl;
        //string out = "Place holder";     
        //Check if sign contains speed or limit        
        if (conf) {
            //int d1 = edit_distance(out, "LIMIT", dis_mat);
            //int d2 = edit_distance(out, "SPEED", dis_mat);
            // Changed to 1.5
            if (!SIGN_TEXT && (edit_distance(out, "LIMIT", dis_mat) <= 1.5 ||
                    edit_distance(out, "SPEED", dis_mat) <= 1.5)) {
                SIGN_TEXT = true;
                TXT_LOC = TxtRegions.at(TxtRegionIdx).Line_Num;
            }

            //Find the speed of the sign (regions are lexicographically sorted)            
            if (SIGN_TEXT) {
                int i;
                std::istringstream(out) >> i; //i is 10 after this                
                if (TxtRegions.at(TxtRegionIdx).Line_Num > TXT_LOC &&
                        i % 5 == 0 && i > 10 && i < 90) {
                    speed = std::to_string(i);
                    loc = TxtRegionIdx;
                    return true;
                }
            }

        }

    }
    return false;
}

/* Levenshtein Distance method
 * From: https://rosettacode.org/wiki/Levenshtein_distance#C.2B.2B
 * Written by: Martin Ettl, 2012-10-05
 */
float Sign_Detector::edit_distance(const std::string& s1, const std::string& s2, float Distance_Mat[36][36]) {
    const std::size_t len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<float>> d(len1 + 1, std::vector<float>(len2 + 1));
    //std::cout<<s1<<" "<<s2<<std::endl;
    d[0][0] = 0;
    for (unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
    for (unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

    for (unsigned int i = 1; i <= len1; ++i)
        for (unsigned int j = 1; j <= len2; ++j) {
            // note that std::min({arg1, arg2, arg3}) works only in C++11,
            // for C++98 use std::min(std::min(arg1, arg2), arg3)

            //Get distance matrix index from ascii values
            int s1_char = s1[i - 1];
            int s2_char = s2[j - 1];
            //std::cout<<len2<<std::endl;
            //set character offsets (for capital letters and numeric 0-9)
            s1_char < 65 ? s1_char -= 22 : s1_char -= 65;
            s2_char < 65 ? s2_char -= 22 : s2_char -= 65;
            float Distance = Distance_Mat[s1_char][s2_char];

            d[i][j] = std::min({d[i - 1][j] + 1, //del
                d[i][j - 1] + 1, //ins
                d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : Distance)}); //swap
        }
    return d[len1][len2];
}

//Sets the distance matrix from a string called filename (should this be in
//FileUtils.h ?)
void Sign_Detector::setDistanceMatrix(std::string filename) {
    std::ifstream file(filename);

    for (int row = 0; row < 36; ++row) {
        std::string line;
        std::getline(file, line);
        if (!file.good())
            break;

        std::stringstream iss(line);

        for (int col = 0; col < 36; ++col) {
            std::string val;
            std::getline(iss, val, ',');
            if (!iss.good())
                break;

            std::stringstream convertor(val);
            convertor >> dis_mat[row][col];
        }
    }
}

bool Sign_Detector::initCascade(std::string CascadeLocAndName) {
    if (!sign_cascade.load(CascadeLocAndName)) {
        printf("Error loading cascade\n");
        return false;
    }

    cascade_init = true;
    return true;
}


/*******************************************************************************
 * ScanImage:
 * Scans an image for a sign and returns an int between and including 1-3
 * 1 - Sign was not found (NO_SIGN_FOUND)
 * 2 - Sign was found AND it's speed read (SIGN_FOUND)
 * 3 - Sign was found AND it's speed WAS NOT read (SIGN_CLASSIFIED)
 * 4 - Special case in which frame should be skipped
 * 
 * inputs:
 * @src         - cv::Mat* representing the image to be scanned for a sign
 * @candidate   - cv::Mat* representing the output region of interest if sign 
 *                is found.
 * Returns: int representing the results of the search
 ******************************************************************************/
int Sign_Detector::ScanImage(cv::Mat &src, cv::Mat &candidate) {

    //set up image for FRST
    cv::Mat scene;
    cv::cvtColor(src, scene, CV_BGR2GRAY);

    //Crop window size            
    //cv::Mat grey_croped(scene, cv::Rect(400, 140, IMG_WIDTH, IMG_HEIGHT));
    cv::Mat grey_croped(scene);
    frst.setImage(grey_croped);

    //Run rectangular FRST
    std::vector<foundShape> Shapes = frst.findRectangles(aspectRatio,
            minWidth, maxWidth, maxShapes, voteThreshold,
            widthStep, angleTolerance);

    //The given voting threshold could cause shapes to be empty: i.e no
    //candidates were found.
    if (Shapes.size() == 0)
        return 4; //special case where the frame needs to be skipped

    //candidate = extractROI(grey_croped, aspectRatio, Shapes.at(0), bb);
    for (int i = 0; i < Shapes.size(); i++) {
        cv::Rect r;
        frst.extractROI(grey_croped, aspectRatio, Shapes[i], r);
        cv::Rect double_r = doubleBB(grey_croped, r);
        //Run cascade classifier on resized image
        Mat roi;
        // Resize image for classifier
        /* TODO: take region from larger image to give better resolution */
        cv::resize(grey_croped(double_r), roi, cv::Size(60, 80));
        sign_cascade.detectMultiScale(roi, CascadeResults);
        if (CascadeResults.size() > 0) // Sign found
        {
            cv::Point old_size = cv::Point(grey_croped(double_r).size().width, grey_croped(double_r).size().height);
            cv::Rect found = extractBBFromParent(double_r, CascadeResults[0], old_size);
            candidate = grey_croped(found);
            bb = found;
            // Toggle here
            bool results = false;            
            if (MODE == MODE_CNN){            
                cv::Mat pyCNN;
                // Resize to appropriate CNN input size
                cv::resize(candidate, pyCNN, cv::Size(28, 28));
                pythonModule.RunCNN(pyCNN);
                int CNN_Results = pythonModule.InterpretResults();
                if (CNN_Results != -1) {
                    speed = std::to_string(CNN_Results);
                    results = true;

                    // Draw box with text for user reference
                    /* TODO: Allow for user toggled detector results */
                    rectangle(src, found, cv::Scalar(0, 255, 0), 2);
                    double fontScale = 2;
                    int thickness = 3;
                    int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
                    cv::putText(src, speed, cv::Point(found.x, found.y), fontFace,
                            fontScale, cv::Scalar(0, 255, 0), thickness, 8);
                                     
                    return SIGN_FOUND;
                }
            }            
            else if (MODE == MODE_OCR) {            
                ScanForText(TxtRegions, candidate, false);
                int loc = 0;
                results = TranslateFoundTxt(TxtRegions,
                        &tesseract_api,
                        loc, OCR_speed);
                
                if (results) {
                    // Draw box with text
                    rectangle(src, found, cv::Scalar(0, 255, 0), 2);
                    double fontScale = 2;
                    int thickness = 3;
                    int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
                    cv::putText(src, speed, cv::Point(found.x, found.y + found.height), fontFace,
                            fontScale, cv::Scalar(255, 0, 0), thickness, 8);
                    return SIGN_FOUND;
                }

            }
                
            return SIGN_CLASSIFIED;

        }

    }

    return NO_SIGN_FOUND;
}

// Run OCR on input image (used for debugging on static images)
int Sign_Detector::RunOCR(cv::Mat img) {
    
    ScanForText(TxtRegions, img, false);
    int loc = 0;
    bool results = TranslateFoundTxt(TxtRegions,
            &tesseract_api,
            loc, speed);
    
    return results;
}

// Run CNN on input image (used for debugging on static images)
int Sign_Detector::RunCNN(cv::Mat img) {
    cv::Mat pyCNN;
    cv::resize(img, pyCNN, cv::Size(28, 28));
    pythonModule.RunCNN(pyCNN);
    int CNN_Results = pythonModule.InterpretResults();
    return CNN_Results;
}

/*******************************************************************************
 * MapRectToParentScale:
 * Takes a rectangle from a scaled down image (child) and maps that rectangle
 * into the scaled up image (parent). This allows for more resolution as the 
 * images are resized to 640x480 from a parent image of 1280x760 or larger.
 * 
 * inputs:
 * @ParentImg         - cv::Mat* representing the larger image
 * @ChildImg          - cv::Mat* representing the smaller image
 * @BB                - cv::Rect* representing the location of the rectangle in
 *                      child image
 * @cropping          -cv::Rect* representing how the child image is cropped
 *                     from it's parent image. For example half of the road is 
 *                     usually croped out since US signs appear on the left hand
 *                     side.
 * Returns: cv::Rect representing the coordinates of child's rectangle when 
 * mapped to the parent
 ******************************************************************************/
cv::Rect Sign_Detector::MapRectToParentScale(cv::Mat &ParentImg, cv::Mat &ChildImg, cv::Rect &BB, cv::Rect &cropping) {
    float x_scale = ((float) ParentImg.size().width) / ChildImg.size().width;
    float y_scale = ((float) ParentImg.size().height) / ChildImg.size().height;

    // Project bounding box origin into scaled image
    int new_x = cropping.x + BB.x;
    int new_y = cropping.y + BB.y;

    // Scale origin
    new_x *= x_scale;
    new_y *= y_scale;

    return cv::Rect(new_x, new_y, x_scale * BB.width, y_scale * BB.height);
}

/* TODO: combine with MapRectToParentScale*/
cv::Rect Sign_Detector::extractBBFromParent(cv::Rect Parent, cv::Rect child, cv::Point old_size) {
    float x_scale = old_size.x / 60.0;
    float y_scale = old_size.y / 80.0;
    //Note child is always strictly contained in the parent rectangle
    int new_x = Parent.x + x_scale * child.x;
    int new_y = Parent.y + y_scale * child.y;

    return cv::Rect(new_x, new_y, x_scale * child.width, y_scale * child.height);
}

// No longer used
int Sign_Detector::constrain(int in, int lowerBoundary, int upperBoundary) {
    if (in > upperBoundary)
        in = upperBoundary;
    if (in < lowerBoundary)
        in = lowerBoundary;
    return in;
}
// Doubles a bouding box r, inside a given image img. Forces the newly
// doubled bounding box is still inside the image.
cv::Rect Sign_Detector::doubleBB(cv::Mat img, cv::Rect r) {
    int h = img.size().height;
    int w = img.size().width;

    int x_new = r.x - r.width / 2;
    int y_new = r.y - r.height / 2;

    int new_width = r.width * 2;
    int new_height = r.height * 2;

    cv::Rect BB2 = cv::Rect(x_new, y_new, new_width, new_height);
    BB2 = cvUtils::constrainRectToImage(img, BB2);
    /*
    x_new = constrain(x_new, 0, w);
    y_new = constrain(y_new, 0, h);
    
    //trim to edges
    if (x_new + new_width > img.size().width)
        new_width = img.size().width - x_new;

    if (y_new + new_height > img.size().height)
        new_height = img.size().height - y_new;
    
    //Force new rectangle to be inside boundaries
     */
    return BB2;

    //return cv::Rect(x_new,y_new,new_width,new_height);

}