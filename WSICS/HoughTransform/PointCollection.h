#ifndef __WSICS_HOUGHTRANSFORM_POINTTRIPLET__
#define __WSICS_HOUGHTRANSFORM_POINTTRIPLET__

#include "Line.h"

namespace WSICS::HoughTransform
{
	/// <summary>
	/// A container class containing three points and three tangents running through these points.
	/// </summary>
    struct PointCollection
    {
		std::vector<std::pair<cv::Point2f, Line>> points;

		/// <summary>
		/// Subtracts a point from the entire collection of points.
		/// </summary>
		/// <param name="subtract_point">The point to subtract from each of the collections points.</param>
		void operator-=(const cv::Point2f& subtract_point);
		/// <summary>
		/// Adds a point to the entire collection of points.
		/// </summary>
		/// <param name="subtract_point">The point to add to each of the collections points.</param>
		void operator+=(const cv::Point2f& addition_point);
    };
}
#endif // __WSICS_HOUGHTRANSFORM_POINTTRIPLET__
