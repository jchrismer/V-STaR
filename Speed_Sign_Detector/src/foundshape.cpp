#include "foundshape.h"

foundShape::foundShape()
{

}

foundShape::foundShape(cv::Point position, int size, int votes)
{
	setPosition(position);
	setSize(size);
	setVotes(votes);
}

cv::Point foundShape::getPosition()
{
	return position;
}

void foundShape::setPosition(cv::Point position)
{
	this->position = position;
}

int foundShape::getSize()
{
	return size;
}

void foundShape::setSize(int size)
{
	this->size = size;
}

int foundShape::getVotes()
{
	return votes;
}

void foundShape::setVotes(int votes)
{
	this->votes = votes;
}
