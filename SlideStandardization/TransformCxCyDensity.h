#ifndef __TRANSFORM_CX_CY_DENSITY_H__
#define __TRANSFORM_CX_CY_DENSITY_H__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "HSD/BackgroundMask.h"
#include "CxCyWeights.h"

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

namespace TransformCxCyDensity
{
	cv::Mat AdjustParamaterMinMax(const cv::Mat& cx_cy, cv::Mat parameters);

	cv::Mat CalculateScaleParameters(std::vector<cv::Point>& indices, cv::Mat& cx_cy_rotated_matrix);

	/// <summary>
	/// Creates seperate matrices for each class and stores all the classification results in one row (vector).
	/// </summary>
	ClassAnnotatedCxCy ClassCxCyGenerator(cv::Mat& all_tissue_classes, cv::Mat& c_x_in, cv::Mat& c_y_in);

	float CovarianceCalculation(cv::Mat& samples_matrix);

	cv::Mat DensityNormalizationThreeScales(ClassDensityRanges& density_ranges,	ClassDensityRanges& lut_density_ranges,	cv::Mat& density_lut, CxCyWeights::Weights& weights);

	ClassPixelIndices GetClassIndices(cv::Mat& all_tissue_classes);
	std::pair<double, double> GetCxCyMedian(cv::Mat& cx_cy_matrix);
	ClassDensityRanges GetDensityRanges(cv::Mat& all_tissue_classes, cv::Mat& Density, ClassPixelIndices& class_pixel_indices);

	std::pair<float, float> GetPercentile(float cx_cy_percentile, cv::Mat& cx_cy);
	std::pair<float, float> GetPercentile(float cx_percentile, float cy_percentile, cv::Mat& cx_cy);

	
	MatrixRotationParameters RotateCxCy(cv::Mat& cx_cy, cv::Mat& output_matrix, cv::Mat& class_cx_cy);
	MatrixRotationParameters RotateCxCy(cv::Mat& cx_cy, cv::Mat& output_matrix, float cx_median, float cy_median, float angle);

	void RotateCxCyBack(cv::Mat& cx_cy_input, cv::Mat& output_matrix, float angle);

	void ScaleCxCy(const cv::Mat& rotated_input, cv::Mat& scaled_output, cv::Mat class_scale_params, cv::Mat& lut_scale_params);
	void ScaleCxCyLUT(const cv::Mat& rotated_input, cv::Mat& scaled_output, cv::Mat class_scale_params, cv::Mat& lut_scale_params);
	void TranslateCxCyBack(cv::Mat& cx_cy_rotated_back_class, cv::Mat& output_matrix, std::vector<cv::Point>& indices, float transform_x_median, float transform_y_median);
};
#endif // __TRANSFORM_CX_CY_DENSITY_H__