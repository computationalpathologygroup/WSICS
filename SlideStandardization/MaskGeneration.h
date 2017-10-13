#ifndef __HE_STAIN_MASK_GENERATION_H__
#define __HE_STAIN_MASK_GENERATION_H__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "HoughTransform/RandomizedHoughTransform.h"
#include "HSD/HSD_Model.h"

namespace HE_Staining
{
	/// <summary>
	/// Contains a mask for the Eosin pixels, as well as a mask that can be used to train a classifier.
	/// </summary>
	struct EosinMaskInformation
	{
		cv::Mat full_mask;
		cv::Mat training_mask;
		size_t training_pixels;
	};

	/// <summary>
	/// Contains a mask for the Hematoxylin pixels, as well as a mask that can be used to train a classifier.
	/// </summary>
	struct HematoxylinMaskInformation
	{
		cv::Mat	full_mask;
		cv::Mat training_mask;
		size_t	training_pixels;
	};

	/// <summary>
	/// Holds methods that can be used to generate masks for Hematoxylin and Eosin masks.
	///
	/// The code Converts the output of the Randomized hough transform algorithm to
	/// a mask. Artifact rejection on the detected ellipses is performed to ensure 
	/// having a a reliable training set for Hematoxylin mask. The Hematoxylin
	/// information is then used to build the Eosin masks.
	/// </summary>
	namespace MaskGeneration
	{
		/// <summary>
		/// Applies a blur to the passed matrix.
		/// </summary>
		/// <param name="matrix">The matrix to blur.</param>
		/// <param name="sigma">The sigma value for the blur transform.</param>
		void ApplyBlur(cv::Mat& matrix, const uint32_t sigma);
		/// <summary>
		/// Applies a Canny Edge transform to the matrix, filtering out unwanted pixels through a threshold operation.
		/// </summary>
		/// <param name="matrix">The matrix to apply the canny edge tranform to.</param>
		/// <param name="low_threshold">The low threshold used to for the threshold operation.</param>
		/// <param name="high_threshold">The high threshold used to for the threshold operation.</param>
		void ApplyCannyEdge(cv::Mat& matrix, const uint32_t low_threshold, const uint32_t high_threshold);
		/// <summary>
		/// Applies a randomized Hough Transform on a binary matrix.
		/// </summary>
		/// <param name="binary_matrix">A binary matrix where each point signifies part of an object. After the transformation, each point will be labeled according to the BLOB they belong to.</param>
		/// <param name="transform_parameters">The parameters used to perform the Hough transform.</param>
		/// <returns>A vector containing the ellipses.</returns>
		std::vector<HoughTransform::Ellipse> ApplyHoughTransform(cv::Mat& binary_matrix, const HoughTransform::RandomizedHoughTransformParameters& transform_parameters);

		/// <summary>
		/// Performs a blur, canny edge and randomized hough transform in order to detect ellipses on the passed matrix.
		/// </summary>
		/// <param name="matrix">The matrix to transform and to acquire ellipses from.</param>
		/// <param name="blur_sigma">The sigma value for the blur transform.</param>
		/// <param name="canny_low_threshold">The low threshold used for the canny edge transform.</param>
		/// <param name="canny_high_threshold">The high threshold used for the canny edge transform.</param>
		/// <returns>A vector containing the ellipses.</returns>
		std::vector<HoughTransform::Ellipse> DetectEllipses(cv::Mat& matrix,
			const uint32_t blur_sigma,
			const uint32_t canny_low_threshold,
			const uint32_t canny_high_threshold,
			const HoughTransform::RandomizedHoughTransformParameters& transform_parameters);

		/// <summary>
		/// Acquires a mean pixel value which is discoverd by selecting the nth element, pointed at by the amount
		/// of pixels * index_percentage in a sorted list.
		/// </summary>
		/// <param name="mean_vector">The mean values of the pixels to select the element from.</param>
		/// <param name="index_percentage">The index percentage used to select the nth element.</param>
		/// <returns></returns>
		double AcquirePercentile(std::vector<float> mean_vector, const float index_percentage);

		/// <summary>
		/// Creates the Eosin mask as well as the corresponding training information. This is done by
		/// inspecting the pixels that haven't been identified as Hematoxylin. The number of training 
		/// samples for the Eosine class are the same as Hema class unless the samples are less than
		/// Hematoxylin.
		/// </summary>
		/// <param name="hsd_image">A HSD format image that corresponds with the Hematoxylin and background masks.</param>
		/// <param name="background_mask">A mask annotating the background pixels.</param>
		/// <param name="hema_training_information">The Hematoxylin mask information.</param>
		/// <param name="eosin_index_percentile">The percentile value, used to select pixels used for the training mask.</param>
		/// <returns>The Eosin mask and training mask.</returns>
		EosinMaskInformation GenerateEosinMasks(
			const HSD::HSD_Model& hsd_image,
			const cv::Mat& background_mask,
			const HematoxylinMaskInformation& hema_training_information,
			const float eosin_index_percentile = 0.85);

		/// <summary>
		/// Creates the Hematoxylin mask as well as the corresponding training information. This is done
		/// after performing artifact filtering, for which the percentile values are used.
		/// </summary>
		/// <param name="hsd_image">A HSD format image that corresponds with the ellipses and background mask.</param>
		/// <param name="background_mask">A mask annotating the background pixels.</param>
		/// <param name="ellipses">The detected ellipses for the image.</param>
		/// <param name="hema_index_percentile">The percentile value, used to select pixels used for the training mask.</param>
		/// <returns>The Hematoxylin mask and training mask.</returns>
		std::pair<bool, HematoxylinMaskInformation> GenerateHematoxylinMasks(
			const HSD::HSD_Model& hsd_image,
			const cv::Mat& background_mask,
			const std::vector<HoughTransform::Ellipse>& ellipses,
			const float hema_index_percentile);

		/// <summary>
		/// Creates a linear vector that holds values calculated by this formula: start_point + i * (end_point - start_point) / (interval - 1).
		/// </summary>
		/// <param name="start_point">The starting point value for the formula..</param>
		/// <param name="end_point">The end point value for the formula.</param>
		/// <param name="interval">The length of the linear vector.</param>
		std::vector<double> LinearSpace(const double start_point, const double end_point, const uint16_t interval);
	}
};
#endif // __HE_STAIN_MASK_GENERATION_H__