#include "Line.h"

#define _USE_MATH_DEFINES

#include <math.h>
#include <stdexcept>

namespace WSICS::HoughTransform
{
    //******************************************************************************
    // Constructors / Destructors
    //******************************************************************************
    Line::Line(void) : theta(0), rho(0)
    {
    }

    Line::Line(const float theta, const float rho) : theta(theta), rho(rho)
    {
    }

	Line::Line(const cv::Point2f& point_a, const cv::Point2f& point_b)
	{
		float height_difference = point_b.x - point_a.x;
		if (height_difference == 0)
		{
			this->rho = point_a.x;
		}

		this->theta = std::atan2(point_b.y - point_a.y, point_b.x - point_a.x);
		if (this->theta < 0)
		{
			this->theta = this->theta + 0.5 * M_PI;
		}
		else
		{
			this->theta = this->theta - 0.5 * M_PI;
		}
		this->rho = (point_a.y * point_b.x - point_a.x * point_b.y) / height_difference * std::sin(this->theta);
	}

	Line::Line(const cv::Point2f& point_a, const std::vector<cv::Point2f>& additional_points)
	{
		// If the point vector only contains a single point, revert to dual point method.
		if (additional_points.size() == 1)
		{
			// TODO: find a more elegant solution.
			Line other(point_a, additional_points[0]);
			this->theta		= other.theta;
			this->rho		= other.rho;
		}

		// Sums the X and Y values of the points vector.
		float sum_x = 0;
		float sum_y = 0;
		float sum_xy = 0;
		float square_x = 0;
		float square_y = 0;
		for (const cv::Point2f& point : additional_points)
		{
			sum_x += point.x;
			sum_y += point.y;
			sum_xy += point.x * point.y;
			square_x += point.x * point.x;
			square_y += point.y * point.y;
		}

		// Calculates the numerator and denominator.
		float numerator = (square_y - std::pow(sum_y, 2) / static_cast<float>(additional_points.size())) - (square_x - std::pow(sum_x, 2) / static_cast<float>(additional_points.size()));
		float denominator = sum_x * sum_y / static_cast<float>(additional_points.size()) - sum_xy;

		float theta = 0;
		float rho = 0;
		if (std::fabs(denominator) < 0.00001f)
		{
			if (numerator < 0) // horizontal
			{
				theta = std::atan2(0.0, 1.0) - 0.5f * M_PI;
				rho = point_a.x * std::cos(theta) + point_a.y * std::sin(theta);
			}
			else if (numerator > 0) // vertical
			{
				theta = std::atan2(1.0, 0.0) - 0.5f * M_PI;
				rho = point_a.x * std::cos(theta) + point_a.y * std::sin(theta);
			}
			else
			{
				theta	= 0;
				rho		= 0;
			}
		}
		else
		{
			float bravo = numerator / denominator / 2.0f;

			float negative_bravo = (-bravo - std::sqrt(bravo * bravo + 1.0f));
			float positive_bravo = (-bravo + std::sqrt(bravo * bravo + 1.0f));

			float negative_alpha = (sum_y - negative_bravo * sum_x) / static_cast<float>(additional_points.size());
			float positive_alpha = (sum_y - positive_bravo * sum_x) / static_cast<float>(additional_points.size());

			float negative_residual_square = 0;
			float positive_residual_square = 0;
			for (const cv::Point2f& point : additional_points)
			{
				negative_residual_square += std::pow((point.y - (negative_alpha + negative_bravo * point.x)), 2);
				positive_residual_square += std::pow((point.y - (positive_alpha + positive_bravo * point.x)), 2);
			}

			if (negative_residual_square < positive_residual_square)
			{
				theta = std::atan(negative_bravo);
				theta = theta + (theta <= 0.0f ? 0.5f * M_PI : -0.5f * M_PI);
				rho = point_a.x * std::cos(theta) + point_a.y * std::sin(theta);
			}
			else
			{
				theta = std::atan(positive_bravo);
				theta = theta + (theta <= 0.0f ? 0.5f * M_PI : -0.5f * M_PI);
				rho = point_a.x * std::cos(theta) + point_a.y * std::sin(theta);
			}
		}

		this->theta		= theta;
		this->rho		= rho;
	}

    //******************************************************************************
    // Public Operators
    //******************************************************************************
    bool Line::operator==(const Line& other) const
    {
        return this->theta == other.theta && this->rho == other.rho;
    }

    bool Line::operator!=(const Line& other) const
    {
        return this->theta != other.theta || this->rho != other.rho;
    }

    //******************************************************************************
    // Public Member Functions
    //******************************************************************************
    float Line::GetAngle(void) const
    {
        float theta = this->theta + (0.5 * M_PI);
        if(std::fabs(theta) > 0.5 * M_PI)
        {
            theta = theta + (theta < 0 ? M_PI : -M_PI);
        }
        return theta;
    }

	cv::Point2f Line::Intersect(const Line& other) const
    {
		float x;
        float y = (other.rho * std::cos(this->theta) - this->rho * std::cos(other.theta)) / std::sin(other.theta - this->theta);
		
        if(std::cos(this->theta) != 0)
        {
            x = (this->rho - y * std::sin(this->theta)) / std::cos(this->theta);
        }
        else
        {
            x = (other.rho - y * std::sin(other.theta)) / std::cos(other.theta);
        }

        return cv::Point2f(x,y);
    }

    bool Line::IsParallelWith(const Line& other) const
    {
		return this->theta == other.theta;
    }

    Line Line::CreateFromSlopeIntercept(const float slope, const float intercept)
    {
        float theta = std::atan(slope) - 0.5 * M_PI;
        float rho	= intercept * std::sin(theta);
        return Line(theta, rho);
    }
}
