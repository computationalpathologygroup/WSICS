#include "Ellipse.h"

#define _USE_MATH_DEFINES

#include <cmath>
#include <math.h> // M_PI

namespace WSICS::HoughTransform
{
	Ellipse::Ellipse(void)
	{
	}

	Ellipse::Ellipse(const cv::Point2f center,  float major_axis, const float minor_axis, const float theta) : center(center), major_axis(major_axis), minor_axis(minor_axis), theta(theta)
	{
	}

	bool Ellipse::operator==(const Ellipse& other) const
	{
		return	this->center			== other.center &&
				this->theta			== other.theta &&
				this->major_axis		== other.major_axis &&
				this->minor_axis		== other.minor_axis;
	}

    bool Ellipse::OnEdge(cv::Point2f point, float const edgeWidth) const 
    {
		point -= this->center;

        float major_axis		= this->major_axis;
        float minor_axis		= this->minor_axis;
        float sinTheta		= std::sin(this->theta);
        float cosTheta		= std::cos(this->theta);

        float outer_major_axis = static_cast<float>(std::pow(major_axis + 0.5 * edgeWidth, 2));
        float outer_minor_axis = static_cast<float>(std::pow(minor_axis + 0.5 * edgeWidth, 2));
        float inner_major_axis = static_cast<float>(std::pow(major_axis - 0.5 * edgeWidth, 2));
        float inner_minor_axis = static_cast<float>(std::pow(minor_axis - 0.5 * edgeWidth, 2));

        float a = std::pow(point.y * sinTheta + point.x * cosTheta, 2);
        float b = std::pow(point.y * cosTheta - point.x * sinTheta, 2);

        float outerResult = a / outer_major_axis + b / outer_minor_axis;
        float innerResult = a / inner_major_axis + b / inner_minor_axis;
        return outerResult <= 1 && innerResult >= 1;
    }

    bool Ellipse::Contains(cv::Point2f point) const
    {
		point -= this->center;

        float sin_theta	= std::sin(this->theta);
        float cos_theta	= std::cos(this->theta);

        float outer_major_axis = std::pow(this->major_axis, 2);
        float outer_minor_axis = std::pow(this->minor_axis, 2);

        float a = std::pow(point.y * sin_theta + point.x * cos_theta, 2);
        float b = std::pow(point.y * cos_theta - point.x * sin_theta, 2);

        return (a / outer_major_axis + b / outer_minor_axis) <= 1;
    }

    size_t Ellipse::GetOverlap(const Ellipse& other_ellipse) const
    {
		cv::Point2f this_top_left(this->center.x		- this->major_axis, this->center.y - this->major_axis);
		cv::Point2f this_top_right(this->center.x		+ this->major_axis, this->center.y - this->major_axis);
		cv::Point2f this_bottom_left(this->center.x	- this->major_axis, this->center.y + this->major_axis);
		cv::Point2f this_bottom_right(this->center.x	+ this->major_axis, this->center.y + this->major_axis);

		cv::Point2f other_top_left(other_ellipse.center.x		- other_ellipse.major_axis, other_ellipse.center.y - other_ellipse.major_axis);
		cv::Point2f other_top_right(other_ellipse.center.x		+ other_ellipse.major_axis, other_ellipse.center.y - other_ellipse.major_axis);
		cv::Point2f other_bottom_left(other_ellipse.center.x		- other_ellipse.major_axis, other_ellipse.center.y + other_ellipse.major_axis);
		cv::Point2f other_bottom_right(other_ellipse.center.x	+ other_ellipse.major_axis, other_ellipse.center.y + other_ellipse.major_axis);

        float left		= 0;
        float right		= 0;
        float top		= 0;
        float bottom		= 0;

        if(SquareContains_(this_top_left, this_bottom_right, other_top_left))
        {
            top		= other_top_left.y;
            left		= other_top_left.x;
        }
        if(SquareContains_(this_top_left, this_bottom_right, other_top_right))
        {
            top		= other_top_right.y;
            right	= other_top_right.x;
        }
        if(SquareContains_(this_top_left, this_bottom_right, other_bottom_left))
        {
            bottom	= other_bottom_left.y;
            left		= other_bottom_left.x;
        }
        if(SquareContains_(this_top_left, this_bottom_right, other_bottom_right))
        {
            bottom	= other_bottom_right.y;
            right	= other_bottom_right.x;
        }

        // reverse
        if(SquareContains_(other_top_left, other_bottom_right, this_top_left))
        {
            top		= this_top_left.y;
            left		= this_top_left.x;
        }
        if(SquareContains_(other_top_left, other_bottom_right, this_top_right))
        {
            top		= this_top_right.y;
            right	= this_top_right.x;
        }
        if(SquareContains_(other_top_left, other_bottom_right, this_bottom_left))
        {
            bottom	= this_bottom_left.y;
            left		= this_bottom_left.x;
        }
        if(SquareContains_(other_top_left, other_bottom_right, this_bottom_right))
        {
            bottom	= this_bottom_right.y;
            right	= this_bottom_right.x;
        }

        if(left == 0 && right == 0 && top == 0 && bottom == 0)
        {
            return 0;
        }
        else
        {
            if(left == 0)
            {
                left = this_top_left.x > other_top_left.x ? this_top_left.x : other_top_left.x;
            }
            if(right == 0)
            {
                right = this_top_right.x < other_top_right.x ? this_top_right.x : other_top_right.x;
            }
            if(top == 0)
            {
                top = this_top_left.y > other_top_left.y ? this_top_left.y : other_top_left.y;
            }
            if(bottom == 0)
            {
                bottom = this_bottom_left.y < other_bottom_left.y ? this_bottom_left.y : other_bottom_left.y;
            }

			size_t count = 0;
            for(size_t x = static_cast<size_t>(left); x <= static_cast<size_t>(right); x++)
            {
                for(size_t y = static_cast<size_t>(top); y <= static_cast<size_t>(bottom); y++)
                {
					cv::Point2f point(x,y);
					if (other_ellipse.Contains(point) && this->Contains(point))
					{
						++count;
					} 
                }
            }
            return count;
        }
    }

	size_t Ellipse::GetSurface(void) const
	{
		size_t count = 0;
		for (int x = (int)(this->center.x - this->major_axis); x <= (int)(this->center.x + this->major_axis); x++)
		{
			for (int y = (int)(this->center.y - this->major_axis ); y <= (int)(this->center.y + this->major_axis); y++)
			{
				if (Contains(cv::Point2f(static_cast<float>(x), static_cast<float>(y))))
				{
					++count;
				}
			}
		}
		return count;
	}

    float Ellipse::GetTangent(const cv::Point2f& point) const
    {
        float a = this->theta		* point.x + this->major_axis * point.y;
        float b = this->major_axis	* point.x + this->minor_axis * point.y;
        float angle = std::atan2(a, -b);
        if(angle > 0.5 * M_PI)
        {
            return angle - M_PI;
        }
        else if(angle < 0.5 * M_PI)
        {
            return angle + M_PI;
        }
        return angle;
    }

	bool Ellipse::SquareContains_(cv::Point2f& a, cv::Point2f& b, cv::Point2f& point) const
	{
		return point.x >= a.x && point.x <= b.x && point.y >= a.y && point.y <= b.y;
	}
}