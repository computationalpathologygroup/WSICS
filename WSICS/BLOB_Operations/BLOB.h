#ifndef __ASAP_IMAGEPROCESSING_BLOBOPERATIONS_BLOB_H
#define __ASAP_IMAGEPROCESSING_BLOBOPERATIONS_BLOB_H

#include <vector>
#include <opencv2/core/types.hpp>

namespace ASAP::Image_Processing::BLOB_Operations
{
	/// <summary>
	///	Represents a BLOB, with several features that aid in image processing.
	///	</summary>
    class BLOB
    {
		public:
			/// <summary>
			/// The standard constructor, creates an empty BLOB.
			/// </summary>
			BLOB(void);
			/// <summary>
			/// Sets the BLOB's top and bottom right corners and then unsafely inserts all the points.
			/// </summary>
			/// <param name="top_left">The top left corner of the BLOB.</param>
			/// <param name="bottom_right">The bottom right corner of the BLOB.</param>
			BLOB(const cv::Point2f& top_left, const cv::Point2f& bottom_right);
			/// <summary>
			/// Initializes the BLOB and unsafely inserts all the points.
			/// </summary>
			/// <param name="blob_points">The points that define the BLOB.</param>
			BLOB(const std::vector<cv::Point2f>& blob_points);
			/// <summary>
			/// Initializes the BLOB's top and bottom right corners and then unsafely inserts all the points.
			/// </summary>
			/// <param name="blob_points">The points that define the BLOB.</param>
			/// <param name="top_left">The top left corner of the BLOB.</param>
			/// <param name="bottom_right">The bottom right corner of the BLOB.</param>
			BLOB(const std::vector<cv::Point2f>& blob_points, const cv::Point2f& top_left, const cv::Point2f& bottom_right);
			/// <summary>
			/// Initializes the BLOB top and bottom right corners and then unsafely moves the point vector.
			/// </summary>
			/// <param name="blob_points">A lvalue vector containing the points that define the BLOB.</param>
			/// <param name="top_left">The top left corner of the BLOB.</param>
			/// <param name="bottom_right">The bottom right corner of the BLOB.</param>
			BLOB(std::vector<cv::Point2f>&& blob_points, const cv::Point2f& top_left, const cv::Point2f& bottom_right);

			/// <summary>
			/// Adds a point to the BLOB, increasing its bounding box if required.
			/// </summary>
			/// <param name="point">The point to include into the BLOB.</param>
			/// <returns>Returns false if the point was already present within the BLOB.</returns>
			bool Add(const cv::Point2f& point);
			/// <summary>
			/// Adds several points to the BLOB, increasing its bounding box if required.
			/// </summary>
			/// <param name="points">The points to include into the BLOB.</param>
			/// <returns>Returns false if any of the points was already present within the BLOB.</returns>
			bool Add(const std::vector<cv::Point2f>& points);
			/// <summary>
			/// Adds the point without performing any checks or increasing the BLOB's bounding box.
			/// </summary>
			/// <param name="point">The point to include into the BLOB.</param>
			void UnsafeAdd(const cv::Point2f& point);
			/// <summary>
			/// Adds the points without performing any checks or increasing the BLOB's bounding box.
			/// </summary>
			/// <param name="points">The points to include into the BLOB.</param>
			void UnsafeAdd(const std::vector<cv::Point2f>& points);

			/// <summary>
			/// Returns a reference to the list of points within the BLOB.
			/// </summary>
			/// <returns>a reference to the list of points.</returns>
			std::vector<cv::Point2f>& GetPoints(void);
			/// <summary>
			/// Returns a constant reference to the list of points within the BLOB.
			/// </summary>
			/// <returns>A constant reference to the list of points.</returns>
			const std::vector<cv::Point2f>& GetPoints(void) const;

			/// <summary>
			/// Returns the upper left point of the BLOB bounding box.
			/// </summary>
			/// <returns>The upper left point of the BLOB bounding box.</returns>
			const cv::Point2f& GetTopLeftPoint(void) const;
			/// <summary>
			/// Returns the bottom right point of the BLOB bounding box.
			/// </summary>
			/// <returns>The bottom right point of the BLOB bounding box.</returns>
			const cv::Point2f& GetBottomRightPoint(void) const;
			/// <summary>
			/// The full height that the BLOB points inhabit.
			/// </summary>
			/// <returns>The height of the area that the BLOB points inhabit.</returns>
			uint32_t GetHeight(void) const;
			/// <summary>
			/// The full width that the BLOB points inhabit.
			/// </summary>
			/// <returns>The width of the area that the BLOB points inhabit.</returns>
			uint32_t GetWidth(void) const;
			/// <summary>
			/// Returns the amount of points or pixels within the BLOB.
			/// </summary>
			/// <returns>The amount of points or pixels within the BLOB.</returns>
			size_t Size(void) const;

			/// <summary>
			/// Returns whether or not there is an intersection of the BLOB bounding boxes.
			/// </summary>
			/// <param name="other">The other BLOB to use for the intersection check.</param>
			/// <returns>Whether or not there is an intersection of the BLOB bounding boxes.</returns>
			bool BoxIntersectsWith(const BLOB& other) const;
			/// <summary>
			/// Returns whether or not there is an intersection between the BLOBs bounding box and the parameter defined one.
			/// </summary>
			/// <param name="top_left">The top left corner of the bounding box.</param>
			/// <param name="bottom_right">The bottom right corner of the bounding box..</param>
			/// <returns>Whether or not there is an intersection between the bounding boxes.</returns>
			bool BoxIntersectsWith(const cv::Point2f& top_left, const cv::Point2f& bottom_right) const;

		private:
			cv::Point2f					m_top_left_;
			cv::Point2f					m_bottom_right_;
			std::vector<cv::Point2f>	m_points_;
    };
}
#endif // __ASAP_IMAGEPROCESSING_BLOBOPERATIONS_BLOB_H