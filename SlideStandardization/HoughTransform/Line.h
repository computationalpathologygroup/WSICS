#ifndef __HOUGHTRANSFORM_LINE_H__
#define __HOUGHTRANSFORM_LINE_H__

#include <opencv2\core\types.hpp>

namespace HoughTransform
{
	/// <summary>
	///	An implementation of a line using an internal representation suitable for
	/// containing vertical lines.
	/// </summary>
    class Line
    {
    public:
		float rho;
		float theta;

		/// <summary>
		/// Default constructor, creates an invalid line.
		/// </summary>
        Line(void);
		/// <summary>
		/// Constructor that initialises the line with the specified theta and rho values.
		/// </summary>
		/// <param name="theta">The theta value for this line.</param>
		/// <param name="rho">The rho value for this line.</param>
		Line(const float theta, const float rho);
		/// <summary>
		/// Constructor that initializes a line based on two points.
		/// </summary>
		/// <param name="point_a">The first point.</param>
		/// <param name="point_b">The second point.</param>
		Line(const cv::Point2f& point_a, const cv::Point2f& point_b);
		/// <summary>
		/// Constructor that initializes a line based on a single point, joined with a vector of points.
		/// </summary>
		/// <param name="point_a">The first point.</param>
		/// <param name="additional_points">A vector with the remaining points.</param>
		Line(const cv::Point2f& point_a, const std::vector<cv::Point2f>& additional_points);

		/// <summary>
		/// Compares the other line to see if they are similar.
		/// </summary>
		/// <param name="other">The other line to compare to this one.</param>
		/// <returns>Whether or not the two lines are similar.</returns>
		bool operator==(const Line& other) const;
		/// <summary>
		/// Compares the other line to see if they are different.
		/// </summary>
		/// <param name="other">The other lien to compare to this one.</param>
		/// <returns>Whether or not the two lines are different.</returns>
		bool operator!=(const Line& other) const;

		/// <summary>
		/// Returns the angle of the line in radians.
		/// </summary>
		/// <returns>The angle of the line in radians.</returns>
        float GetAngle(void);
		/// <summary>
		/// Gets the point of intersection between this and the other line.
		/// </summary>
		/// <param name="other">The other line to calculate the intersection with.</param>
		/// <returns>The point of intersection between the two lines.</returns>
		cv::Point2f Intersect(const Line& other);
		/// <summary>
		/// Returns whether or not the two lines run in parallel.
		/// </summary>
		/// <param name="other">The other line to compare with.</param>
		/// <returns>Whether or not the lines run in parallel.</returns>
        bool IsParallelWith(const Line& other);

		/// <summary>
		/// Creates a line from a slope and interception value.
		/// </summary>
		/// <param name="slope">The slope of the line.</param>
		/// <param name="intercept">The interception of the line.</param>
		/// <returns>A line based on slope and interception.</returns>
        static Line CreateFromSlopeIntercept(const float slope, const float intercept);
    };
}
#endif // __HOUGHTRANSFORM_LINE_H__
