#include "FRST.h"

FRST::FRST()
{
	this->callCounter = 0;
	this->seenBeforeCheck = false;        
}

FRST::FRST(Mat image)
{
	this->image = image;
	this->callCounter = 0;
	this->seenBeforeCheck = false;
}

void FRST::setImage(Mat image)
{
	this->image = image;
}



std::vector<foundShape> FRST::findRectangles(cv::Size aspectRatio, int minWidth, 
        int maxWidth, int maxShapes, int voteThreshold, int widthStep, int angleTolerance)
{            
	std::vector<foundShape> bestShapes;
	std::vector<Mat> voteImages;
	std::vector<int> sizes;
        
        /******* MODIFICATIONS ************* 
         Find gradient once, rather than each loop update
         */
        Mat xSquared(image.rows, image.cols, CV_32F);
	Mat ySquared(image.rows, image.cols, CV_32F);	
        
        // Compute gradient magnitudes
	cv::Sobel(image, xSobel, CV_32F, 1, 0);     //xSobel and ySobel are the only changes
	cv::Sobel(image, ySobel, CV_32F, 0, 1);
	cv::multiply(xSobel, xSobel, xSquared);
	cv::multiply(ySobel, ySobel, ySquared);
	magnitudes = xSquared+ySquared;
	cv::sqrt(magnitudes, magnitudes);

        Mat sorted(image.rows, image.cols, CV_32F);

	// Find the threshold for the top 20%. Only done for each 10 frames, since it is slow:
	if(callCounter++ % 10 == 0) {
		sorted = magnitudes.clone();
		sorted = sorted.reshape(0, 1);
		cv::sort(sorted, sorted, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);
		currentThreshold = sorted.at<float>(0, floor(0.9f*sorted.cols));
	}        
	// Vote for the desired radii of the shape:       
	for(int width = minWidth; width <= maxWidth; width += widthStep) {
		voteImages.push_back(voteRectangle(aspectRatio, width, angleTolerance));
		sizes.push_back(width);
	}
        //Above is slowest part of function taking around 0.03 second to finish
        
        
	findShapes(&bestShapes, voteImages, sizes, maxShapes, voteThreshold);   //fast
	callCounter++;
	return bestShapes;
}

void FRST::findGradients(float* thresholdValue)
{	
	Mat sorted(image.rows, image.cols, CV_32F);

	// Find the threshold for the top 20%. Only done for each 10 frames, since it is slow:
	if(callCounter++ % 10 == 0) {
		sorted = magnitudes.clone();
		sorted = sorted.reshape(0, 1);
		cv::sort(sorted, sorted, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);
		currentThreshold = sorted.at<float>(0, floor(0.9f*sorted.cols));
	}

	*thresholdValue = currentThreshold;

}

Mat FRST::voteRectangle(cv::Size aspectRatio, int width, int angleTolerance)
{	
        // image.size => Voting image remains the same size
	Mat votes(image.size(), CV_32S, cv::Scalar(0));
	float thresholdValue = currentThreshold;

	int height = cvUtils::round(((float)aspectRatio.height/(float)aspectRatio.width)*width);

	// Find the acceptable grandient ranges:
	std::vector<cv::Range> ranges;
	ranges.push_back(cv::Range(-angleTolerance, angleTolerance)); // Vertical lines (horizontal gradient)
	ranges.push_back(cv::Range(180-angleTolerance, 180+angleTolerance)); // Vertical lines
	ranges.push_back(cv::Range(90-angleTolerance, 90+angleTolerance)); // Horizontal lines (vertical gradient)
	ranges.push_back(cv::Range(270-angleTolerance, 270+angleTolerance)); // Horizontal lines

	cv::Vec2f unitGradient;
	cv::Point currentPoint;
	float angle;

	// Loop through gradients:
	for(currentPoint.y = 0; currentPoint.y < magnitudes.rows; currentPoint.y++) {            
		for(currentPoint.x = 0; currentPoint.x < magnitudes.cols; currentPoint.x++) {                    
			// Ignore gradients below the threshold:
			if(magnitudes.at<float>(currentPoint.y, currentPoint.x) < thresholdValue) {
				continue;
			}

			//temp1.at<float>(currentPoint.y, currentPoint.x) = magnitudes.at<float>(currentPoint.y, currentPoint.x);

			// Ignore gradients not in accepted angular ranges:
			angle = cv::fastAtan2(ySobel.at<float>(currentPoint.y, currentPoint.x), xSobel.at<float>(currentPoint.y, currentPoint.x));
			if(!cvUtils::inRanges(angle, ranges)) {
				continue;
			}

			//temp2.at<float>(currentPoint.y, currentPoint.x) = magnitudes.at<float>(currentPoint.y, currentPoint.x);

			float length = std::sqrt(std::pow(xSobel.at<float>(currentPoint.y, currentPoint.x), 2) + std::pow(ySobel.at<float>(currentPoint.y, currentPoint.x), 2));
			unitGradient[0] = xSobel.at<float>(currentPoint.y, currentPoint.x)/length;
			unitGradient[1] = ySobel.at<float>(currentPoint.y, currentPoint.x)/length;

			if(angle < 45) {
				vote(votes, unitGradient, currentPoint, width/2, height/2); // Vote horizontal
			} else {
				vote(votes, unitGradient, currentPoint, height/2, width/2); // Vote vertical
			}
		}
	}

	return votes;
}

void FRST::findShapes(std::vector<foundShape>* bestShapes, std::vector<Mat> voteImages, std::vector<int> sizes, int maxShapes, int voteThreshold)
{
	cv::Point point;
	Mat totalVotes(voteImages[0].size(), CV_32S, cv::Scalar(0));
	Mat mask(voteImages[0].size(), CV_8U, cv::Scalar(255));
	std::vector<foundShape> checkedShapes;
	int maxVotes = 0, bestSize = 0;
	double accVotes;
        
        // Vote image based on the total number of votes found
	for(unsigned int i = 0; i < voteImages.size(); i++) {
		totalVotes += voteImages[i];
	}

	for(int i = 0; i < maxShapes; i++) {
                //src,minVal,maxVal,MinLoc,Maxloc,mask
		cv::minMaxLoc(totalVotes, NULL, &accVotes, NULL, &point, mask);
                // pixel location of the largest vote is stored in point
		
                maxVotes = 0;
                //Go through vote images at the max point to find the max Votes.
		for(unsigned int j = 0; j < voteImages.size(); j++) {
			if(voteImages[j].at<int>(point) > maxVotes) {
				maxVotes = voteImages[j].at<int>(point);
                                //Rectangle width of best (highest) votes
				bestSize = sizes[j];
			}
		}

		if(maxVotes < voteThreshold) { // There are no points left above the threshold.
			break;
		}

                /* 
                * At the maximum point draw a black circle of width equal to the width
                * corresponding to the highest voted position. The circle nulls
                * the area out and acts as the increment to the loop, ie.
                * decreasing the search area of minMaxLoc, providing an exit.
                * Since the radius is bestSize, the circle is inside the rect.
                */
		circle(mask, point, bestSize, cv::Scalar(0), -1);
		
		if(seenBeforeCheck) { // Prepare for the seen-before-check in the next round by adding all potential shapes to the list.
			checkedShapes.push_back(foundShape(point, bestSize, maxVotes));
		}

		if(seenBeforeCheck && !seenBefore(foundShape(point, bestSize, maxVotes))) { // If this shape was not in the previous frame, we don't believe it's a legitimate shape (provided we actually care about that).
			continue;
		}

		// All tests are passed, so this is one of the best shapes. Let's go add it.
		bestShapes->push_back(foundShape(point, bestSize, maxVotes));

	}

	previousFindings = checkedShapes;
}

bool FRST::seenBefore(foundShape shape, int allowedShift)
{
	for(unsigned int i = 0; i < previousFindings.size(); i++) {
		if(shape.getPosition().x > previousFindings[i].getPosition().x-allowedShift
				&& shape.getPosition().x < previousFindings[i].getPosition().x+allowedShift
				&& shape.getPosition().y > previousFindings[i].getPosition().y-allowedShift
				&& shape.getPosition().y < previousFindings[i].getPosition().y+allowedShift) {
			return true;
		}
	}

	return false;
}

void FRST::angleRanges(std::vector<cv::Range>* ranges, int numSides, int rotation, int angleTolerance)
{
	cv::Range range;
	int baseAngle;
	for(int i = 0; i < numSides; i++) {
		baseAngle = cvUtils::round((float)i*(360.0f/(float)numSides)+90+rotation); // The base angle is rotated by 90 degrees, since the gradients are orthogonal to the sides of the shape.
		range.start = (baseAngle-angleTolerance)%360;
		range.end = (baseAngle+angleTolerance)%360;
		ranges->push_back(range);
                
	}
}

// To do: Threading/Parallel processing for speed?
void FRST::vote(Mat votes, cv::Vec2f unitGradient, cv::Point basePoint, int w, int distance)
{
	cv::Point votePoint;
	int vote;

	// Voting mechanism:
	for(int m = -2*w; m <= 2*w; m++) {
		vote = (m >= -w && m <= w) ? 1 : -1; // Vote positive when within -w and w. Otherwise, vote negative.

		// First we vote in the gradient direction (as per the + sign in the round()-call):
		votePoint.x = basePoint.x + cvUtils::round(m*-unitGradient[1] + distance*unitGradient[0]);
		votePoint.y = basePoint.y + cvUtils::round(m*unitGradient[0] + distance*unitGradient[1]);
		if(cvUtils::inImage(votePoint, votes.size())) { // Make sure the vote is inside the image.			
                        votes.at<int>(votePoint.y,votePoint.x) += vote;                        
		}

		// Then we vote in the opposite direction:
		votePoint.x = basePoint.x + cvUtils::round(m*-unitGradient[1] - distance*unitGradient[0]);
		votePoint.y = basePoint.y + cvUtils::round(m*unitGradient[0] - distance*unitGradient[1]);
		if(cvUtils::inImage(votePoint, votes.size())) { // Make sure the vote is inside the image.                        
			votes.at<int>(votePoint.y,votePoint.x) += vote;                        
		}
	}
}

void FRST::extractROI(cv::Mat image, cv::Size aspectRatio, foundShape bestShapes, cv::Rect &BB) {
    
    int height = cvUtils::round(bestShapes.getSize()*((float) aspectRatio.width / (float) aspectRatio.height));
    int width = bestShapes.getSize();

    cv::Point upperLeft = cv::Point(bestShapes.getPosition().x - width / 2, bestShapes.getPosition().y - height / 2);    
    cv::Rect r(upperLeft.x, upperLeft.y, width, height);
    
    //force valid regions
    if (r.x < 0) {
        //Save call to abs()
        r.width = r.width + r.x;
        r.x = 0;
    }

    if (r.y < 0) {
        //Save call to abs()
        r.height = r.height + r.y;
        r.y = 0;
    }
    //trim to edges
    if (r.x + r.width > image.size().width)
        r.width = image.size().width - r.x;

    if (r.y + r.height > image.size().height)
        r.height = image.size().height - r.y;
    BB = r;    
}

bool FRST::getSeenBefore()
{
	return seenBeforeCheck;
}

void FRST::setSeenBefore(bool seenBefore)
{
	this->seenBeforeCheck = seenBefore;
}