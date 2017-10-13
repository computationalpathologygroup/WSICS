#ifndef __HOUGHTRANSFORM_WINDOWEDTRIPLETDETECTOR_H__
#define __HOUGHTRANSFORM_WINDOWEDTRIPLETDETECTOR_H__

#include <vector>
#include <unordered_set>

#include "../BLOB_Operations/BLOB_Window.h"
#include "Ellipse.h"
#include "PointCollection.h"

namespace HoughTransform
{
	/// <summary>
	/// Defines the types of point selection for the WindowedTripletDetector.
	/// </summary>
    enum PointSelection
    {
        POINT_SELECTION_FULLY_LABELED					= 1,
        POINT_SELECTION_PARTIALLY_LABELED				= 2,
        POINT_SELECTION_FULLY_RANDOM					= 3,
        POINT_SELECTION_FULLY_LABELED_DIST_RESTR		= 4,
        POINT_SELECTION_PARTIALLY_LABELED_DIST_RESTR	= 5,
        POINT_SELECTION_FULLY_RANDOM_DIST_RESTR			= 6
    };

	/// <summary>
	/// Holds all the parameters that define the WindowedTripletDetector its execution.
	/// </summary>
	struct WindowedTripletDetectorParameters
	{
		float			ellipse_range_tolerance;
		float			ellipse_removal_range;
		float			min_coverage;
		float			min_point_distance;
		float			max_point_distance;
		float			tangent_search_radius;
		uint32_t			window_size;
		PointSelection	point_selection;
	};

	/// <summary>
	/// Divides a binary matrix into subwindows and then retrieves triplets on it.
	/// </summary>
    class WindowedTripletDetector
    {
		public:
			WindowedTripletDetectorParameters parameters;

			/// <summary>
			/// Default constructs the WindowedTripletDetector, using standard parameters.
			/// </summary>
			WindowedTripletDetector(void);
			/// <summary>
			/// Constructs the WindowedTripletDetector with all the information it requires to proccess a matrix.
			/// </summary>
			/// <param name="parameters">Holds the parameters that are used to define the execution of the WindowTripletDetector.</param>
			WindowedTripletDetector(const WindowedTripletDetectorParameters parameters);

			/// <summary>
			/// Initializes the detector for processing by transforming a binary matrix into a labeled BLOB matrix.
			/// </summary>
			/// <param name="binary_matrix">The binary matrix to transform.</param>
			/// <param name="mask_type">The type of mask used to detect the BLOBs.</param>
			void Initialize(cv::Mat& binary_matrix, const ASAP::Image_Processing::BLOB_Operations::MaskType mask_type);
			/// <summary>
			/// Initializes the detector for processing by recovering the labeled blobs from the matrix.
			/// </summary>
			/// <param name="labeled_blob_matrix">The matrix holding the labeled BLOBs.</param>
			/// <param name="stats_array">An array holding the label statistics.</param>
			void Initialize(cv::Mat& labeled_blob_matrix, cv::Mat& stats_array);
			/// <summary>
			/// Checks if this object has been initialized for processing. False could indicate that the initalize method
			/// hasn't been called, or the passed matrix contained no BLOB's that could be acquired.
			/// </summary>
			/// <returns>Whether or not the object has been initialized.</returns>
			bool IsInitialized(void);
			/// <summary>
			/// Acquires a triplet from the current window, based on the current settings.
			/// 
			/// The initial point of a triplet will always be selected fully random, the additional two will
			/// either be fully random as well, fully random within a BLOB or based on the range around / 
			/// within a BLOB.
			/// </summary>
			/// <returns>A PointCollection with three entries, defining a triplet.</returns>
			PointCollection GetNextTriplet(void);
			void Clear(void);
			/// <summary>
			/// Shifts the window to the next region, updating its information to reflect such.
			/// </summary>
			/// <return>Returns false if the window can no longer be shifted and has reached the end of the matrix.</returns>
			bool Next(void);
			/// <summary>
			/// Reverts the window to the initial position, updating its internal state to reflect such.
			/// </summary>
			void Reset(void);
			/// <summary>
			/// Reduces the amount of points around the passed ellipse.
			/// </summary>
			/// <param name="ellipse">The elipse around which to search and delete points.</param>
			void Simplify(const Ellipse& ellipse);
			/// <summary>
			/// Returns the current amount of labeled points within the windowed region of the matrix..
			/// </summary>
			/// <returns>The current amount of labeled points within the window.</returns>
			size_t Size(void);
			/// <summary>
			/// Verification is done by creating two virtual ellipses based on the ellipse to verify, their radii are enlarged and decreased
			/// by the "ellipse_range_tolerance" variable. The current list of labeled points is then iterated over to see how many points fall 
			/// between the virtual ellipses. These are the points that are considered part of the ellipse.
			///
			/// The total amount of points is then normalized by using the amount of pixels located on the edge of the ellipse's bounding box.
			/// The resulting value is compared against the "min_coverage" to approve or disprove it.
			/// </summary>
			/// <param name="ellipse">The ellipse to verify.</param>
			/// <returns>Whether or not the ellipse is valid.</returns>
			bool Verify(const Ellipse& ellipse);

			/// <summary>
			/// Returns the standard parameters for the WindowedTripletDetector.
			/// </summary>
			/// <returns>The standard parameters for the WindowedTripletDetector.</returns>
			static WindowedTripletDetectorParameters GetStandardParameters(void);

		protected:
			/// <summary>
			/// Acquires a random labeled point from the current window.
			/// </summary>
			/// <returns>A pairing with the BLOB label and a pointer towards the point.</returns>
			inline std::pair<size_t, cv::Point2f*> GetRandomLabeledPoint$(void);
			/// <summary>
			/// Acquires a random labeled point from a BLOB within the window.
			/// </summary>
			/// <param name="label">The label of the BLOB to select the point from.</param>
			/// <returns>A pairing with the BLOB label and a pointer towards the point.</returns>
			inline std::pair<size_t, cv::Point2f*> GetRandomLabeledPoint$(size_t label);

		private:
			std::unordered_map<size_t, ASAP::Image_Processing::BLOB_Operations::BLOB*>	m_labeled_blobs_;
			std::unordered_set<cv::Point2f*>												m_deleted_points_;
			std::unordered_map<size_t, std::vector<cv::Point2f*>>						m_labeled_points_;

			ASAP::Image_Processing::BLOB_Operations::BLOB_Window							m_blob_window_;
			size_t																		m_total_window_points_;

			/// <summary>
			/// Acquires a triplet fully randomly.
			/// </summary>
			/// <param name="a_from_same_label">Whether or not Alpha should be selected from the origin label.</param>
			/// <param name="b_from_same_label">Whether or not Bravo should be selected from the origin label.</param>
			/// <param name="labeled_origin">The origin point for the triplet.</param>
			/// <returns>A PointCollection with three entries, defining a triplet.</returns>
			PointCollection AcquireRandomTriplet_(const bool a_from_same_label, const bool b_from_same_label, std::pair<size_t, cv::Point2f*>& labeled_origin);
			/// <summary>
			/// Acquires a triplet where each point is within the max and min distance of each other.
			/// </summary>
			/// <param name="a_from_same_label">Whether or not Alpha should be selected from the origin label.</param>
			/// <param name="b_from_same_label">Whether or not Bravo should be selected from the origin label.</param>
			/// <param name="labeled_origin">The origin point for the triplet.</param>
			/// <returns>A PointCollection with three entries, defining a triplet.</returns>
			PointCollection AcquireRangeRestrictedTriplet_(const bool a_from_same_label, const bool b_from_same_label, std::pair<size_t, cv::Point2f*>& labeled_origin);

			/// <summary>
			/// Calculates the tangent of a point, based on its member labled BLOB points.
			/// </summary>
			/// <param name="point">The point to base the tangent on.</param>
			/// <returns>The tangent as a line object.</returns>
			Line CalculateTangent_(std::pair<size_t, cv::Point2f*>& point);

			/// <summary>
			/// Checks if the object has been properly initialized, throws an exception on failure.
			/// </summary>
			void CheckValidAccess_(void);

			/// <summary>
			/// Acquires points within the max radius around center and not within the min.
			/// </summary>
			/// <param name="center">The center point around which to project the radii.</param>
			/// <returns>A vector containing the points that fit the criteria</returns>
			std::vector<std::pair<size_t, cv::Point2f*>> GetPointsFromRadius_(const cv::Point2f& center);
			/// <summary>
			/// Acquires points from a single label within the max radius around center and not within the min.
			/// </summary>
			/// <param name="center">The center point around which to project the radii.</param>
			/// <param name="label">The label attached to the BLOB from which the points should be selected.</param>
			/// <returns>A vector containing the points that fit the criteria.</returns>
			std::vector<std::pair<size_t, cv::Point2f*>> GetPointsFromRadius_(const cv::Point2f& center, const size_t label);
			/// <summary>
			/// Returns whether or not the position is located within the radius around the center.
			/// </summary>
			/// <param name="center">The center point around which the radius will be projected.</param>
			/// <param name="position">The position to check against the center radius.</param>
			/// <param name="radius">The radius to use for the calculation.</param>
			/// <returns>Whether or not point is within the radius around center.</returns>
			bool IsPositionedWithinRadius_(const cv::Point& center, cv::Point2f& position, const float radius);
			/// <summary>
			/// Rebuilds the window information based on the current state of the object.
			/// </summary>
			void UpdateWindowInformation_(void);
    };
}
#endif // __HOUGHTRANSFORM_WINDOWEDTRIPLETDETECTOR_H__
