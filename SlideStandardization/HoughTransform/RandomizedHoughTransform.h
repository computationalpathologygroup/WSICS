#ifndef __HOUGHTRANSFORM_RANDOMIZEDHOUGHTRANSFORM_H__
#define __HOUGHTRANSFORM_RANDOMIZEDHOUGHTRANSFORM_H__

#include "TreeAccumulator.h"
#include "WindowedTripletDetector.h"

// TODO: Convert stack of variables into structs that can be passed down to the corresponding objects.

namespace HoughTransform
{
    enum EllipseRemoval
    {
        ELLIPSE_REMOVAL_NONE = 0,
        ELLIPSE_REMOVAL_SIMPLE = 1,
        ELLIPSE_REMOVAL_EMPTY_ACCUMULATOR = 2,
        ELLIPSE_REMOVAL_ALL_IN_EPOCH = 3,
        ELLIPSE_REMOVAL_BEST_IN_EPOCH = 4
    };

    enum MidpointCalculation
    {
        MIDPOINT_CALCULATION_DEFAULT = 0,
        MIDPOINT_CALCULATION_OPTIMAL = 1,
    };

    enum CombineThreshold
    {
        COMBINE_THRESHOLD_IDENTITY = 1,
        COMBINE_THRESHOLD_FACTOR2 = 2,
        COMBINE_THRESHOLD_FACTOR3 = 3
    };

    enum TangentVerification
    {
        TANGENT_VERIFICATION_NONE = 0,
        TANGENT_VERIFICATION_SINGLE = 1,
        TANGENT_VERIFICATION_DOUBLE = 2,
        TANGENT_VERIFICATION_TRIPLE = 3
    };

	struct RandomizedHoughTransformParameters
	{
		size_t				count_threshold;
		float				ellipse_position_threshold;
		float				ellipse_radii_threshold;
		float				epoch_size;
		float				min_ellipse_radius;
		float				max_ellipse_radius;
		float				tangent_tolerance;
		MidpointCalculation	midpoint_calculation;
		EllipseRemoval		ellipse_removal_method;
		CombineThreshold	combine_threshold;
		TangentVerification tangent_verification;
	};

	/// <summary>
	/// Provides a restricted randomized Hough transform.
	/// </summary>
    class RandomizedHoughTransform
    {
		public:
			RandomizedHoughTransformParameters parameters;

			/// <summary>
			/// Constructs the RandomizedHoughTransform based with standard parameters.
			/// </summary>
			RandomizedHoughTransform(void);
			/// <summary>
			/// Constructs the RandomizedHoughTransform based on the passed parameters.
			/// </summary>
			/// <param name="parameters">The parameters that define the execution of the Hough Transform.</param>
			RandomizedHoughTransform(RandomizedHoughTransformParameters parameters);
			/// <summary>
			/// Constructs the RandomizedHoughTransform and sets the WindowedTripletCreator parameters.
			/// </summary>
			/// <param name="hough_transform_parameters">The parameters that define the execution of the Hough Transform.</param>
			/// <param name="triplet_detector_parameters">The parameters that define the execution of the windowed triplet detector.</param>
			RandomizedHoughTransform(RandomizedHoughTransformParameters hough_transform_parameters, WindowedTripletDetectorParameters triplet_detector_parameters);

			/// <summary>
			/// Performs ellipse detection on a binary matrix.
			/// </summary>
			/// <param name="binary_matrix">A binary matrix where each true point signifies part of an object.</param>
			/// <param name="output_matrix">The matrix that will hold the BLOB labeling results.</param>
			/// <param name="mask_type">The type of mask to use for the BLOB searching.</param>
			/// <returns>The detected ellipses.<returns>
			std::vector<Ellipse> Execute(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const ASAP::Image_Processing::BLOB_Operations::MaskType mask_type);
			/// <summary>
			/// Performs ellipse detection on a labeled BLOBs matrix.
			/// </summary>
			/// <param name="labeled_matrix">A matrix containing labeled BLOBs.</param>
			/// <param name="stats_array">The corresponding statistics for each BLOB.</param>
			/// <returns>The detected ellipses.<returns>
			std::vector<Ellipse> Execute(const cv::Mat& labeled_matrix, const cv::Mat& stats_array);
			/// <summary>
			/// Performs ellipse detection on points made available through a WindowedTripletDetector.
			/// </summary>
			/// <param name="triplet_detector">A fully initialized WindowedTripletDetector to apply the Hough transform onto.</param>
			/// <returns>The detected ellipses.<returns>
			std::vector<Ellipse> Execute(WindowedTripletDetector& triplet_detector);

			/// <summary>
			/// Returns the standard parameters for the RandomizedHoughTransform.
			/// </summary>
			/// <returns>The standard parameters for the RandomizedHoughTransform.</returns>
			static RandomizedHoughTransformParameters GetStandardParameters(void);

		private:
			WindowedTripletDetectorParameters m_triplet_detector_parameters_;

			/// <summary>
			/// Computers the a,b,c parameters for an ellipse that has a location which is already known. It
			/// does this by using a point triplet. Application of linear decomposition then determines the 
			/// three values that would define an ellipse, with its center located on the origin.
			/// </summary>
			/// <param name="triplet">The triplet to use as the basis for the computation.</param>
			/// <param name="ellipse">The ellipse which already has a location.</param>
			/// <returns>Whether or not the new ellipse values are valid.</returns>
			bool ComputeParameters_(PointCollection& triplet, Ellipse& ellipse);
			/// <summary>
			/// Converts the ellipses parameters from an a, b, c format to a theta, major axis, minor axis format.
			/// The reason for doing so is to create a more accumulator friendly range of numbers for each parameter.
			/// </summary>
			/// <param name="ellipse">The ellipse to convert.</param>
			void ConvertParameters_(Ellipse& ellipse) const;
			/// <summary>
			/// Determines the center of an ellipse through the usage of a point triplet and its tangents. This is achieved
			/// by making use of the symmetry of an ellipse.
			/// </summary>
			/// <param name="triplet">The triplet from which to determine its center.</param>
			/// <returns>The center point of the triplet.</returns>
			cv::Point2f DetermineCenter_(PointCollection& triplet);
			/// <summary>
			/// Attempts to create a new ellipse through a point triplet. It performs several checks
			/// in order to report the validity of the created ellipse.
			/// </summary>
			/// <param name="triplet">The point triplet to convert into an ellipse.</param>
			/// <returns>A pair where the first value defines whether or not the Ellipse in the second value is valid.</returns>
			std::pair<bool, Ellipse> EllipseFromTriplet_(PointCollection& triplet);

			/// <summary>
			/// Determines whether an ellipse its size falls into a certain range. To accomplish this, the variables "max_ellipse_radius"
			/// and "min_ellipse_radius" are used. If an ellipse is too big or too small, "false" will be returned.
			/// </summary>
			/// <param name="ellipse">The ellipse to check.</param>
			/// <returns>Whether or not the ellipse has a valid radius.</returns>
			bool HasCorrectRadius_(const Ellipse& ellipse) const;
			/// <summary>
			/// Determines whether an ellipse defines contours that intersect with the three points of the triplet. It does this by
			/// calculating the tangent at each of the three points, using the ellipses parameters. It then compares those with the
			/// tangets calculated by the triplet itself. If the difference between the tangents is too large, "false" will be returned.
			/// </summary>
			/// <param name="triplet">The triplet to match to the Ellipses contours.</param>
			/// <param name="ellipse">The ellipse that defines the contours.</param>
			/// <returns>Whether or not the ellipse and triplet contorus align within a certain range.</returns>
			bool FitsContours_(PointCollection& triplet, const Ellipse& ellipse) const;
			/// <summary>
			/// Performs the neccesary operations for the ellipse extraction.
			/// </summary>
			/// <param name="ellipse">The detected ellipse.</param>
			/// <param name="best_ellipse">The currently best performing ellipse, based on the amount of detections.</param>
			/// <param name="best_ellipse_count">The highest amount of counts for the detected ellipses.</param>
			/// <param name="repeat_epoch">Whether or not to repeat the current epoch.</param>
			/// <param name="accumulator">The accumulator for this epoch.</param>
			/// <param name="combiner">The accumulator for the entire execution.</param>
			/// <param name="triplet_detector">The triplet detector which is providing the points for the detection.</param>
			bool ProcessDetectedEllipse_(Ellipse& ellipse,
				Ellipse& best_ellipse,
				size_t& best_ellipse_count,
				bool& repeat_epoch,
				TreeAccumulator& accumulator,
				TreeAccumulator& combiner,
				WindowedTripletDetector& triplet_detector);
    };
}
#endif