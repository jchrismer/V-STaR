#ifndef FOUNDSHAPE_H
#define FOUNDSHAPE_H

#include <opencv2/core/core.hpp>

typedef cv::Mat Mat;

class foundShape
{
public:
    foundShape();
	foundShape(cv::Point position, int size, int votes);

	cv::Point getPosition();
	void setPosition(cv::Point position);
	int getSize();
	void setSize(int size);
	int getVotes();
	void setVotes(int votes);

private:
	cv::Point position;
	int size;
	int votes;
};

#endif // FOUNDSHAPE_H
