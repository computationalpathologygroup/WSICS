#include "PointCollection.h"

namespace HoughTransform
{
    //******************************************************************************
    // Public Operators
    //******************************************************************************
    void PointCollection::operator-=(const cv::Point2f& subtract_point)
    {
		for (std::pair<cv::Point2f, Line>& point : points)
		{
			point.first.x -= subtract_point.x;
			point.first.y -= subtract_point.y;
		}
    }

    void PointCollection::operator+=(const cv::Point2f& addition_point)
    {
		for (std::pair<cv::Point2f, Line>& point : points)
		{
			point.first.x += addition_point.x;
			point.first.y += addition_point.y;
		}
    }
}
