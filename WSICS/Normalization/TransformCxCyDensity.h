#ifndef __WSICS_NORMALIZATION_TRANSFORMCXCYDENSITY__
#define __WSICS_NORMALIZATION_TRANSFORMCXCYDENSITY__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../HSD/BackgroundMask.h"
#include "CxCyWeights.h"

namespace WSICS::Normalization::TransformCxCyDensity
{
	struct MedianPercentileValues
	{
		double hema_median_cx;
		double hema_median_cy;
		double eosin_median_cx;
		double eosin_median_cy;
	};

	struct ClassAnnotatedCxCy
	{
		cv::Mat cx_cy_merged;
		cv::Mat hema_cx_cy;
		cv::Mat eosin_cx_cy;
		cv::Mat background_cx_cy;
	};

	struct ClassDensityRanges
	{
		cv::Scalar hema_density_mean;
		cv::Scalar hema_density_standard_deviation;
		cv::Scalar eosin_density_mean;
		cv::Scalar eosin_density_standard_deviation;
		cv::Scalar background_density_mean;
		cv::Scalar background_density_standard_deviation;
	};

	struct ClassPixelIndices
	{
		std::vector<cv::Point> hema_indices;
		std::vector<cv::Point> eosin_indices;
		std::vector<cv::Point> background_indices;
	};

	struct MatrixRotationParameters
	{
		float	angle;
		float	x_median;
		float	y_median;
	};

	cv::Mat AdjustParamaterMinMax(const cv::Mat& cx_cy, cv::Mat parameters);

	cv::Mat CalculateScaleParameters(const std::vector<cv::Point>& indices, const cv::Mat& cx_cy_rotated_matrix);

	/// <summary>
	/// Creates seperate matrices for each class and stores all the classification results in one row (vector).
	/// </summary>
	ClassAnnotatedCxCy ClassCxCyGenerator(const cv::Mat& all_tissue_classes, const cv::Mat& cx_cy_in);

	float CovarianceCalculation(const cv::Mat& samples_matrix);

	cv::Mat DensityNormalizationThreeScales(const ClassDensityRanges& density_ranges, const ClassDensityRanges& lut_density_ranges, const cv::Mat& density_lut, const CxCyWeights::Weights& weights);

	ClassPixelIndices GetClassIndices(const cv::Mat& all_tissue_classes);
	std::pair<double, double> GetCxCyMedian(const cv::Mat& cx_cy_matrix);
	ClassDensityRanges GetDensityRanges(const cv::Mat& all_tissue_classes, const cv::Mat& Density, const ClassPixelIndices& class_pixel_indices);

	std::pair<float, float> GetPercentile(const float cx_cy_percentile, const cv::Mat& cx_cy);
	std::pair<float, float> GetPercentile(const float cx_percentile, const float cy_percentile, const cv::Mat& cx_cy);

	
	MatrixRotationParameters RotateCxCy(const cv::Mat& cx_cy, cv::Mat& output_matrix, const  cv::Mat& class_cx_cy);
	MatrixRotationParameters RotateCxCy(const cv::Mat& cx_cy, cv::Mat& output_matrix, const float cx_median, const float cy_median, const float angle);

	void RotateCxCyBack(const cv::Mat& cx_cy_input, cv::Mat& output_matrix, const float angle);

	void ScaleCxCy(const cv::Mat& rotated_input, cv::Mat& scaled_output, const cv::Mat& class_scale_params, const cv::Mat& lut_scale_params);
	void ScaleCxCyLUT(const cv::Mat& rotated_input, cv::Mat& scaled_output, const cv::Mat& class_scale_params, const cv::Mat& lut_scale_params);

	void TranslateCxCyBack(const cv::Mat& cx_cy, const cv::Mat& cx_cy_lut, cv::Mat& output_matrix, const std::vector<cv::Point>& indices, const float transform_x_median, const float transform_y_median);
};
#endif // __WSICS_NORMALIZATION_TRANSFORMCXCYDENSITY__