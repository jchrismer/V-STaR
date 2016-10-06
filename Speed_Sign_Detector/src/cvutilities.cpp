#include "cvutilities.h"

#include "stdio.h"

namespace cvUtils {

cv::Rect UsrRect;
bool bDraw;
cv::Rect r;
cv::Point base;

Mat img;
Mat layer;
Mat working;
Mat CropedImg;

cv::Rect constrainRectToImage(cv::Mat &Img, cv::Rect &r)
{
    cv::Rect constrained = r;
    
    // Trim negatives
    if (r.x < 0) {        
        constrained.width = r.width + r.x;
        constrained.x = 0;
    }

    if (r.y < 0) {        
        constrained.height = r.height + r.y;
        constrained.y = 0;
    }
    
    // Trim to edges
    if (r.x + r.width > Img.size().width)
        constrained.width = Img.size().width - r.x;

    if (r.y + r.height > Img.size().height)
        constrained.height = Img.size().height - r.y;
    
    return constrained;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    
    if ( event == cv::EVENT_LBUTTONDOWN )
    {        
        // Init your rect
        base.x = x;
        base.y = y;
        r.x = x;
        r.y = y;
        r.width = 0;
        r.height = 0;
        bDraw = true;
    }
    
    else if ( event == cv::EVENT_MOUSEMOVE )
    {

        // If drawing, update rect width and height
        if(!bDraw) return;

        int dx = abs(r.x - x);
        int dy = abs(r.y - y);

        if(x < base.x) {
            r.x = x;
            r.width = abs(x - base.x);
        } else {
            r.width = dx;
        }

        if(y < base.y) {
            r.y = y;
            r.height = abs(y - base.y);
        } else {
            r.height = dy;
        }

        // Refresh        
        working = layer.clone();
        cv::rectangle(working, r, cv::Scalar(0,255,0));
        cv::imshow("Crop new area", working);
    }
    
    else if ( event == cv::EVENT_LBUTTONUP)
    {

        // Save rect, draw it on layer
        UsrRect = r;        

        r = cv::Rect(); 
        bDraw = false;

        // Refresh
        CropedImg = layer(UsrRect);
        working = layer.clone();
        cv::rectangle(working, UsrRect, cv::Scalar(0,255,255),2);
        //cv::rectangle(working, r, cv::Scalar(0,255,0));
        cv::imshow("Results", CropedImg);
        cv::imshow("Crop new area", working);
    }
}

void UsrCrop(Mat &in, cv::Rect &usrSel)
{   
    
    // initialize your temp images
    img = in.clone();
    layer = img.clone();
    working = img.clone();

    // Must create a window to attach the callback to
    cv::namedWindow("Crop new area", 1);

    // Set the user crop callback to window
    cv::setMouseCallback("Crop new area", CallBackFunc, NULL);

    //show the image
    imshow("Crop new area", working);

    // Wait until user presses 'q'
    while(true)
    {
        char key = cv::waitKey(0);
        if(key == 'q')
        {
            cv::destroyWindow("Crop new area");
            cv::destroyWindow("Results");
            return;
        }
        
        if(key == 's')
        {
            usrSel = UsrRect;
            cv::destroyWindow("Crop new area");
            cv::destroyWindow("Results");
            return;
        }
    };    
}

float round(float d)
{
	return floor(d + 0.5);
}

bool inImage(cv::Point point, cv::Size imageSize)
{
	return point.x >= 0 && point.x < imageSize.width && point.y >=0 && point.y < imageSize.height;
}

bool inRanges(float value, std::vector<cv::Range> ranges)
{
	for(unsigned int i = 0; i < ranges.size(); i++) {
		if((ranges[i].start <= ranges[i].end && (value >= ranges[i].start && value < ranges[i].end)) || (ranges[i].start > ranges[i].end && (value >= ranges[i].start || value < ranges[i].end)) ) {
			//printf("%f in range.\n", value);
			return true;
		}
	}
	//printf("%f not in any accepted range.\n", value);
	return false;
}

void saveOverlay(const char* filename, Mat img1, Mat img2, float blending)
{
	Mat temp(img1.size(), CV_8UC1);

	cv::MatIterator_<uchar>tempIt = temp.begin<uchar>();
	cv::MatIterator_<uchar>img1It = img1.begin<uchar>();
	cv::MatIterator_<uchar>img2It = img2.begin<uchar>();
	for(; tempIt != temp.end<uchar>(); ++tempIt, ++img1It, ++img2It) {
		*tempIt = *img1It*(1-blending)+*img2It*blending;
	}
	cv::imwrite(filename, temp);
}

/**
 * Draw a regular polygon
 *
 * Draw a regular, closed polygon.
 * @param image the image to draw in.
 * @param numSides the number of sides the polygon should have.
 * @param radius radius of the circumscribed circle.
 * @param center the center of the polygon.
 * @param rotation the rotation of the polygon.
 * @param color the line color.
 * @param thickness the width of the line.
 */
void drawRegPolygon(Mat image, int numSides, float radius, cv::Point center, float rotation, cv::Scalar color, int thickness)
{
	cv::Point vertices[numSides];
	for(int i = 0; i < numSides; i++) {
		vertices[i].x = center.x + radius * cos(2*M_PI*i/numSides+rotation*M_PI/180);
		vertices[i].y = center.y +radius * sin(2*M_PI*i/numSides+rotation*M_PI/180);
	}
	const cv::Point* curveArr[1] = {vertices};

	cv::polylines(image, curveArr, &numSides, 1, true, color, thickness);
}

/**
 * Add a number to all matrix elements, with wrap-around.
 *
 * Negative numbers can be added. Works only with CV_8UC1 matrices.
 */
void hueAdd(Mat image, int value, Mat mask, int upperBound) {
	assert(image.type() == CV_8UC1);
	assert(mask.type() == CV_8UC1);

	for(int i = 0; i < image.rows; i++) {
		for(int j = 0; j < image.cols; j++) {
			if(mask.at<uchar>(i,j) == 0) {
				continue;
			}

			int result = (int)image.at<uchar>(i, j) + value; // Typecasting to make sure negative results can be saved.

			if(result > upperBound) {
				image.at<uchar>(i, j) = result-upperBound;
			} else if(result < 0) {
				image.at<uchar>(i, j) = upperBound+result;
			} else {
				image.at<uchar>(i, j) = result;
			}

		}
	}
}

void warpPerspective( const Mat& src, Mat& dst, const Mat& M0, cv::Size dsize, int flags, int borderType, const cv::Scalar& borderValue, cv::Point origin)
{
	dst.create(dsize, src.type());

	const int BLOCK_SZ = 32; // Block-size.
	short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];
	double M[9];
	Mat _M(3, 3, CV_64F, M);
	int interpolation = flags & cv::INTER_MAX;
	if(interpolation == cv::INTER_AREA) {
		interpolation = cv::INTER_LINEAR;
	}

	CV_Assert((M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 && M0.cols == 3);
	M0.convertTo(_M, _M.type());

	if(!(flags & cv::WARP_INVERSE_MAP)) {
		invert(_M, _M);
	}

	int x, xDest, y, yDest, x1, y1, width = dst.cols, height = dst.rows;

	// Calculate the sizes of the blocks the image will be split into (bh is block height, bw is block width):
	int bh0 = std::min(BLOCK_SZ/2, height);
	int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, width);
	bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, height);

	// Loop through the blocks:
	for(y = -origin.y, yDest = 0; y < height; y += bh0, yDest += bh0) {
		for(x = -origin.x, xDest = 0; x < width; x += bw0, xDest += bw0) {

			// Find the size of the current block - either the normal size, or smaller, if the block lies near the edge of the image:
			int bw = std::min(bw0, width - xDest);
			int bh = std::min(bh0, height - yDest);

			// Avoid dimension errors:
			if (bw <= 0 || bh <= 0) {
				break;
			}

			Mat _XY(bh, bw, CV_16SC2, XY); // The map for use in remap.
			Mat dpart(dst, cv::Rect(xDest, yDest, bw, bh)); // The destination ROI for this block.

			// Loop through each row of the current block (and subsequently each pixel to calculate the map):
			for(y1 = 0; y1 < bh; y1++) {
				short* xy = XY + y1*bw*2; // Pointer to the first slot in the current row of the map.

				// Calculate the transformation (a simple matrix-vector product):
				double X0 = M[0]*x + M[1]*(y + y1) + M[2];
				double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
				double W0 = M[6]*x + M[7]*(y + y1) + M[8];

				if(interpolation == cv::INTER_NEAREST) {
					// Loop through each column in the current block-row:
					for(x1 = 0; x1 < bw; x1++)
					{
						double W = W0 + M[6]*x1;
						W = W ? 1./W : 0;
						int X = cv::saturate_cast<int>((X0 + M[0]*x1)*W);
						int Y = cv::saturate_cast<int>((Y0 + M[3]*x1)*W);
						xy[x1*2] = (short)X;
						xy[x1*2+1] = (short)Y;
					}
				} else {
					short* alpha = A + y1*bw;
					// Loop through each column in the current block-row:
					for(x1 = 0; x1 < bw; x1++) {
						double W = W0 + M[6]*x1;
						W = W ? cv::INTER_TAB_SIZE/W : 0;
						int X = cv::saturate_cast<int>((X0 + M[0]*x1)*W);
						int Y = cv::saturate_cast<int>((Y0 + M[3]*x1)*W);
						xy[x1*2] = (short)(X >> cv::INTER_BITS) + origin.x;
						xy[x1*2+1] = (short)(Y >> cv::INTER_BITS) + origin.y;
						alpha[x1] = (short)((Y & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (X & (cv::INTER_TAB_SIZE-1)));
					}
				}
			}

			// Remap using the calculated maps:
			if(interpolation == cv::INTER_NEAREST) {
				remap(src, dpart, _XY, Mat(), interpolation, borderType, borderValue);
			} else {
				Mat _A(bh, bw, CV_16U, A);
				remap(src, dpart, _XY, _A, interpolation, borderType, borderValue);
			}
		}
	}
}

Mat resizeCanvas(Mat input, cv::Scalar fill, int left, int right, int up, int down)
{
	assert(input.rows+up+down > 0 && input.cols+right+left > 0);

	Mat newCanvas(input.rows+up+down, input.cols+left+right, input.type(), fill);

	int leftOffset = std::max(0, left);
	int rightOffset = std::min(0, right);
	int upOffset = std::max(0, up);
	int downOffset = std::min(0, down);
	int inputLeftOffset = (int)std::abs(std::min(0, left));
	int inputUpOffset = (int)std::abs(std::min(0, up));

	int roiWidth = std::min(newCanvas.cols, input.cols+rightOffset);
	int roiHeight = std::min(newCanvas.rows, input.rows+downOffset);

	//std::cout << "canvasROI: " << leftOffset << " " << upOffset << " " << roiWidth << " " << roiHeight << " (img size: " << newCanvas.cols << "x" << newCanvas.rows << ")" << std::endl;
	Mat canvasROI = newCanvas(cv::Rect(leftOffset, upOffset, roiWidth, roiHeight));

	//std::cout << "inputROI: " << inputLeftOffset << " " << inputUpOffset << " " << roiWidth << " " << roiHeight << " (img size: " << newCanvas.cols << "x" << newCanvas.rows << ")" << std::endl;
	Mat inputROI = input(cv::Rect(inputLeftOffset, inputUpOffset, roiWidth, roiHeight));

	inputROI.copyTo(canvasROI);
	return newCanvas;
}

Mat resizeCanvas(Mat input, cv::Scalar fill, int* parameters)
{
	return resizeCanvas(input, fill, parameters[0], parameters[1], parameters[2], parameters[3]);
}

Mat autoCrop(Mat input, int* parameters)
{
	assert(input.depth() == CV_8U);
	assert(input.channels() == 3 || input.channels() == 1);

	int minX = input.cols, minY = input.rows, maxX = 0, maxY = 0;

	if(input.channels() == 3) {
		cv::Vec3b baseColor = input.at<cv::Vec3b>(0,0);
		for(int y = 0; y < input.rows; y++) {
			for(int x = 0; x < input.cols; x++) {
				if(input.at<cv::Vec3b>(y,x)[0] != baseColor[0] || input.at<cv::Vec3b>(y,x)[1] != baseColor[1] || input.at<cv::Vec3b>(y,x)[2] != baseColor[2]) {
					minX = x < minX ? x : minX;
					minY = y < minY ? y : minY;
					maxX = x > maxX ? x : maxX;
					maxY = y > maxY ? y : maxY;
				}
			}
		}
	} else {
		uchar baseColor = input.at<uchar>(0,0);
		for(int y = 0; y < input.rows; y++) {
			for(int x = 0; x < input.cols; x++) {
				if(input.at<uchar>(y,x) != baseColor) {
					minX = x < minX ? x : minX;
					minY = y < minY ? y : minY;
					maxX = x > maxX ? x : maxX;
					maxY = y > maxY ? y : maxY;
				}
			}
		}
	}

	if(parameters != NULL) {
		parameters[0] = -minX;
		parameters[1] = maxX-input.cols;
		parameters[2] = -minY;
		parameters[3] = maxY-input.rows;
	}

	return resizeCanvas(input, cv::Scalar(0), -minX, maxX-input.cols, -minY, maxY-input.rows);
}

Mat blendMask(Mat foreground, Mat background, Mat inputMask)
{
	assert(foreground.type() == CV_8UC3 && background.type() == CV_8UC3);
	assert(inputMask.type() == CV_8UC1);
	assert(foreground.cols == background.cols && foreground.rows == background.rows && foreground.rows == inputMask.rows && foreground.cols == inputMask.cols);

	std::vector<Mat> fgChannels, bgChannels;
	Mat mask(foreground.rows, foreground.cols, CV_32F);
	Mat inverseMask(foreground.rows, foreground.cols, CV_32F);
	Mat tempForeground, tempBackground;

	inputMask.convertTo(mask, CV_32F);
	mask = mask/255.0;
	inverseMask = 1-mask;

	foreground.convertTo(tempForeground, CV_32FC3);
	background.convertTo(tempBackground, CV_32FC3);

	split(tempForeground, fgChannels);
	split(tempBackground, bgChannels);
	for(unsigned int i = 0; i < fgChannels.size(); i++) {
		fgChannels[i] = fgChannels[i].mul(mask);
		bgChannels[i] = bgChannels[i].mul(inverseMask);
	}
	merge(fgChannels, tempForeground);
	merge(bgChannels, tempBackground);

	tempForeground.convertTo(tempForeground, CV_8UC3);
	tempBackground.convertTo(tempBackground, CV_8UC3);

	return tempForeground+tempBackground;
}

}
