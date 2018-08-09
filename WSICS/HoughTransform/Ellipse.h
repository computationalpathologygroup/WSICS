#ifndef __HOUGHTRANSFORM_ELLIPSE_H__
#define __HOUGHTRANSFORM_ELLIPSE_H__

#include <opencv2/core/types.hpp>

namespace HoughTransform
{
	/// <summary>
	/// A container class containing the parameters that describe an ellipse.
	///	</summary>
    struct Ellipse
    {
		public:
			/// <summary>The center of an ellipse.</summary>
			cv::Point2f	center;
			/// <summary>The length of the large-axis of an ellipse.</summary>
			float		major_axis;
			/// <summary>The length of the short-axis of an ellipse.</summary>
			float		minor_axis;
			/// <summary>The amount of rotation of an ellipse.</summary>
			float		theta;

			/// <summary>
			/// Constructs an empty ellipse.
			/// </summary>
			Ellipse(void);

			/// <summary>
			/// Constructs the ellipse.
			/// </summary>
			/// <param name="center">The origin of the ellipse.</param>
			/// <param name="major_axis">The size of the large or long axis of the ellipse.</param>
			/// <param name="minor_axis">The size of the minor or short axis of the ellipse.</param>
			/// <param name="theta">The rotation of the ellipse.</param>
			Ellipse(const cv::Point2f center, const float major_axis, const float minor_axis, const float theta);

			/// <summary>
			/// Compares whether or not the two ellipses are identical.
			///	</summary>
			/// <param name="other">The other ellipse to compare to this.</param>
			/// <returns>Whether or not the two ellipses are identical.</returns>
			bool operator==(const Ellipse& other) const;

			/// <summary>
			/// Checks whether a point is contained by the contour of the ellipse.
			///	</summary>
			/// <param name="point">A point that is either contained by the ellipse it's contour or not.</param>
			/// <returns>A value indicating whether or not the point is contained by the contour of the ellipse.</returns>
			bool Contains(cv::Point2f point) const;

			/// <summary>
			/// Returns the size of the overlap between this and the other ellipse.
			/// 	</summary>
			/// <param name="ellipse">The other ellipse to calculate the overlap with.</param>
			/// <returns>The overlap between this and the passed ellipse.</returns>
			size_t GetOverlap(const Ellipse& ellipse) const;
			/// <summary>
			/// Gets the surface area size of the ellipse.
			///	</summary>
			/// <returns>The surface area size of the ellipse.</returns>
			size_t GetSurface(void) const;
			/// <summary>
			/// Calculates the tangent of the ellipse at the specified point.
			///	</summary>
			/// <param name="point">The point for which to calculate the tangent.</param>
			/// <returns>The tangent of the ellipse at the specified point.</returns>
			float GetTangent(const cv::Point2f& point) const;
			/// <summary>
			/// Checks whether or not a point is located on the edge of the ellipse.
			///	</summary>
			/// <param name="point">A point that is either on the edge of the ellipse or not.</param>
			/// <param name="edgeWidth">The width of the ellipe it's edge.</param>
			/// <returns>A value indicating whether or not the point is on the edge of the ellipse.</returns>
			bool OnEdge(cv::Point2f point, const float edgeWidth) const;

	private:
			/// <summary>
			/// Checks whether or not the rectangular area described by the two points contains the third point.
			/// </summary>
			/// <param name="a">Upper left corner.</param>
			/// <param name="b">Bottom right corner.</param>
			/// <param name="point">The point to compare to the rectangle.</param>
			/// <returns>Whether or not the point is placed within the rectangle.</returns>
			inline bool SquareContains_(cv::Point2f& a, cv::Point2f& b, cv::Point2f& point) const;
    };
}
#endif // __HOUGHTRANSFORM_ELLIPSE_H__
