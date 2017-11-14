#include "TransformCxCyDensity.h"

#define _USE_MATH_DEFINES

#include <math.h>
#include "MiscMatrixOperations.h"

namespace TransformCxCyDensity
{
	cv::Mat AdjustParamaterMinMax(const cv::Mat& cx_cy, cv::Mat parameters)
	{
		double min, max;
		cv::minMaxIdx(cx_cy.col(0), &min, &max);
		parameters.at<float>(0, 0)	= min;
		parameters.at<float>(6, 0)	= max;

		cv::minMaxIdx(cx_cy.col(1), &min, &max);
		parameters.at<float>(0, 1)	= min;
		parameters.at<float>(6, 1)	= max;

		return parameters;
	}

	cv::Mat CalculateScaleParameters(std::vector<cv::Point>& indices, cv::Mat& cx_cy_rotated_matrix)
	{
		cv::Mat cx_cy_values = cv::Mat::zeros(indices.size(), 2, CV_32FC1);

		for (size_t i = 0; i < indices.size(); ++i)
		{
			cx_cy_values.at<float>(i, 0) = cx_cy_rotated_matrix.at<float>(indices[i].y, 0);
			cx_cy_values.at<float>(i, 1) = cx_cy_rotated_matrix.at<float>(indices[i].y, 1);
		}

		cv::Mat sorted_cx_cy_values;
		cv::sortIdx(cx_cy_values, sorted_cx_cy_values, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

		std::vector<double> mins(2);
		std::vector<double> maxs(2);
		cv::minMaxIdx(cx_cy_rotated_matrix.col(0), &mins[0], &maxs[0]);
		cv::minMaxIdx(cx_cy_rotated_matrix.col(1), &mins[1], &maxs[1]);

		cv::Mat cx_cy_params(cv::Mat::zeros(7, 2, CV_32F));
		for (size_t col = 0; col < cx_cy_params.cols; ++col)
		{
			cx_cy_params.at<float>(0, col) = mins[col];
			cx_cy_params.at<float>(0, col) = cx_cy_values.at<float>(sorted_cx_cy_values.at<int>(sorted_cx_cy_values.rows * 0.01, col), col);
			cx_cy_params.at<float>(0, col) = cx_cy_values.at<float>(sorted_cx_cy_values.at<int>(sorted_cx_cy_values.rows * 0.25, col), col);
			cx_cy_params.at<float>(0, col) = cx_cy_values.at<float>(sorted_cx_cy_values.at<int>(sorted_cx_cy_values.rows * 0.50, col), col);
			cx_cy_params.at<float>(0, col) = cx_cy_values.at<float>(sorted_cx_cy_values.at<int>(sorted_cx_cy_values.rows * 0.75, col), col);
			cx_cy_params.at<float>(0, col) = cx_cy_values.at<float>(sorted_cx_cy_values.at<int>(sorted_cx_cy_values.rows * 0.99, col), col);
			cx_cy_params.at<float>(0, col) = maxs[col];
		}

		return cx_cy_params;
	}

	ClassAnnotatedCxCy ClassCxCyGenerator(cv::Mat& all_tissue_classes, cv::Mat& c_x_in, cv::Mat& c_y_in)
	{
		cv::Mat all_tissue_classes_uchar;
		all_tissue_classes.convertTo(all_tissue_classes_uchar, CV_8UC1);

		size_t cx_cy_position = 0;
		cv::Mat cx_cy(cv::Mat::zeros(all_tissue_classes.rows * all_tissue_classes.cols, 2, CV_32FC1));
		cv::Mat all_tissues_classes_vector(cv::Mat::zeros(all_tissue_classes.rows * all_tissue_classes.cols, 1, CV_8UC1));

		// Points to each pixel value in turn assuming a CV_8UC1 greyscale image 
		std::vector<cv::Point> hema_pixels;
		std::vector<cv::Point> eosin_pixels;
		std::vector<cv::Point> background_pixels;
		for (uint32_t row = 0; row < all_tissue_classes_uchar.rows; ++row)
		{
			for (uint32_t col = 0; col < all_tissue_classes_uchar.cols; ++col)
			{
				switch (all_tissue_classes_uchar.at<uchar>(row, col))
				{
					case 1: hema_pixels.push_back(cv::Point(row, col));			break;
					case 2: eosin_pixels.push_back(cv::Point(row, col));		break;
					case 3: background_pixels.push_back(cv::Point(row, col));	break;
				}

				cx_cy.at<float>(cx_cy_position, 0) = c_x_in.at<float>(row, col);
				cx_cy.at<float>(cx_cy_position, 1) = c_y_in.at<float>(row, col);
				++cx_cy_position;
			}
		}

		ClassAnnotatedCxCy class_annotated_cx_cy
		{
			cx_cy,
			cv::Mat::zeros(hema_pixels.size(), 2, CV_32FC1),
			cv::Mat::zeros(eosin_pixels.size(), 2, CV_32FC1),
			cv::Mat::zeros(background_pixels.size(), 2, CV_32FC1)
		};

		for (size_t pixel = 0; pixel < hema_pixels.size(); ++pixel)
		{
			class_annotated_cx_cy.hema_cx_cy.at<float>(pixel, 0) = c_x_in.at<float>(hema_pixels[pixel]);
			class_annotated_cx_cy.hema_cx_cy.at<float>(pixel, 1) = c_y_in.at<float>(hema_pixels[pixel]);
		}
		for (size_t pixel = 0; pixel < eosin_pixels.size(); ++pixel)
		{
			class_annotated_cx_cy.eosin_cx_cy.at<float>(pixel, 0) = c_x_in.at<float>(eosin_pixels[pixel]);
			class_annotated_cx_cy.eosin_cx_cy.at<float>(pixel, 1) = c_y_in.at<float>(eosin_pixels[pixel]);
		}
		for (size_t pixel = 0; pixel < background_pixels.size(); ++pixel)
		{
			class_annotated_cx_cy.background_cx_cy.at<float>(pixel, 0) = c_x_in.at<float>(background_pixels[pixel]);
			class_annotated_cx_cy.background_cx_cy.at<float>(pixel, 1) = c_y_in.at<float>(background_pixels[pixel]);
		}

		return class_annotated_cx_cy;
	}

	float CovarianceCalculation(cv::Mat& samples_matrix)
	{
		cv::Mat covariance_matrix;
		cv::Mat mean_matrix;

		cv::calcCovarMatrix(samples_matrix, covariance_matrix, mean_matrix, CV_COVAR_NORMAL | CV_COVAR_ROWS);
		covariance_matrix /= (samples_matrix.rows - 1);
		covariance_matrix.convertTo(covariance_matrix, CV_32FC1);

		cv::Mat eigen_input(2, 2, CV_32FC1);
		eigen_input.at<float>(0, 0) = covariance_matrix.at<float>(0, 0);
		eigen_input.at<float>(0, 1) = covariance_matrix.at<float>(0, 1);
		eigen_input.at<float>(1, 0) = covariance_matrix.at<float>(1, 0);
		eigen_input.at<float>(1, 1) = covariance_matrix.at<float>(1, 1);
		cv::Mat eigen_value_output;
		cv::Mat eigen_vector_output;
		cv::eigen(eigen_input, eigen_value_output, eigen_vector_output);

		float angle = eigen_vector_output.at<float>(1, 1);
		if (angle <= M_PI / 2)
		{
			angle = M_PI - angle;
		}
		return angle;
	}

	cv::Mat DensityNormalizationThreeScales(ClassDensityRanges& class_density_ranges, ClassDensityRanges& lut_density_ranges, cv::Mat& density_lut, CxCyWeights::Weights& weights)
	{
		float std_ratio_hema = class_density_ranges.hema_density_standard_deviation[0] / lut_density_ranges.hema_density_standard_deviation[0];
		float std_ratio_eosin = class_density_ranges.eosin_density_standard_deviation[0] / lut_density_ranges.eosin_density_standard_deviation[0];
		float std_ratio_background = class_density_ranges.background_density_standard_deviation[0] / lut_density_ranges.background_density_standard_deviation[0];

		cv::Mat lut_hema_density_normalized			= cv::Mat::zeros(density_lut.rows, 1, CV_32FC1);
		cv::Mat lut_eosin_density_normalized		= cv::Mat::zeros(density_lut.rows, 1, CV_32FC1);
		cv::Mat lut_background_density_normalized	= cv::Mat::zeros(density_lut.rows, 1, CV_32FC1);
		for (size_t row = 0; row < density_lut.rows; ++row)
		{
			float density_value = density_lut.at<float>(row, 0);
			lut_hema_density_normalized.at<float>(row, 0) = (density_value - class_density_ranges.hema_density_mean[0]) * std_ratio_hema + lut_density_ranges.hema_density_mean[0];
			lut_eosin_density_normalized.at<float>(row, 0) = (density_value - class_density_ranges.eosin_density_mean[0]) * std_ratio_eosin + lut_density_ranges.eosin_density_mean[0];
			lut_background_density_normalized.at<float>(row, 0) = (density_value - class_density_ranges.background_density_mean[0]) * std_ratio_background + lut_density_ranges.background_density_mean[0];
		}

		cv::Mat scaled_density_matrix(cv::Mat::zeros(density_lut.rows, 1, CV_32FC1));

		size_t counter = 0;
		for (size_t row = 0; row < weights.hema.rows; ++row)
		{
			for (size_t col = 0; col < weights.hema.cols; ++col)
			{
				scaled_density_matrix.at<float>(counter, 0) = lut_hema_density_normalized.at<float>(counter, 0)		* weights.hema.at<float>(row, col) +
					lut_eosin_density_normalized.at<float>(counter, 0)		* weights.eosin.at<float>(row, col) +
					lut_background_density_normalized.at<float>(counter, 0) * weights.background.at<float>(row, col);
				++counter;
			}
		}

		return scaled_density_matrix;
	}

	ClassPixelIndices GetClassIndices(cv::Mat& all_tissue_classes)
	{
		size_t full_size = all_tissue_classes.rows * all_tissue_classes.cols;
		cv::Mat hema_mask = cv::Mat::zeros(full_size, 1, CV_8UC1);
		cv::Mat eosin_mask = cv::Mat::zeros(full_size, 1, CV_8UC1);
		cv::Mat background_mask = cv::Mat::zeros(full_size, 1, CV_8UC1);

		size_t hema_count = 0;
		size_t eosin_count = 0;
		size_t background_count = 0;

		for (size_t row = 0; row < all_tissue_classes.rows; ++row)
		{
			for (size_t col = 0; col < all_tissue_classes.cols; ++col)
			{
				switch ((uchar)all_tissue_classes.at<float>(row, col))
				{
					case 1: hema_mask.at<uchar>(hema_count, 0)				= 1; ++hema_count;			break;
					case 2: eosin_mask.at<uchar>(eosin_count, 0)			= 1; ++eosin_count;			break;
					case 3: background_mask.at<uchar>(background_count, 0)	= 1; ++background_count;	break;
				}
			}
		}

		ClassPixelIndices class_pixel_indices;
		cv::findNonZero(hema_mask, class_pixel_indices.hema_indices);
		cv::findNonZero(eosin_mask, class_pixel_indices.eosin_indices);
		cv::findNonZero(background_mask, class_pixel_indices.background_indices);

		return class_pixel_indices;
	}

	std::pair<double, double> GetCxCyMedian(cv::Mat& cx_cy_matrix)
	{
		cv::Mat sorted_cx_cy;
		cv::sortIdx(cx_cy_matrix, sorted_cx_cy, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

		return
		{
			cx_cy_matrix.at<float>(sorted_cx_cy.at<int>(sorted_cx_cy.rows * 0.50, 0), 0),
			cx_cy_matrix.at<float>(sorted_cx_cy.at<int>(sorted_cx_cy.rows * 0.50, 1), 1)
		};
	}

	ClassDensityRanges GetDensityRanges(cv::Mat& all_tissue_classes, cv::Mat& Density, ClassPixelIndices& class_pixel_indices)
	{
		cv::Mat hema_density(cv::Mat::zeros(class_pixel_indices.hema_indices.size(), 1, CV_32FC1));
		cv::Mat eosin_density(cv::Mat::zeros(class_pixel_indices.eosin_indices.size(), 1, CV_32FC1));
		cv::Mat background_density(cv::Mat::zeros(class_pixel_indices.background_indices.size(), 1, CV_32FC1));

		size_t hema_count = 0;
		size_t eosin_count = 0;
		size_t background_count = 0;
		for (size_t row = 0; row < all_tissue_classes.rows; ++row)
		{
			for (size_t col = 0; col < all_tissue_classes.cols; ++col)
			{
				switch ((uchar)all_tissue_classes.at<float>(row, col))
				{
					case 1: hema_density.at<float>(hema_count, 0)				= Density.at<float>(row, col);  ++hema_count;		break;
					case 2: eosin_density.at<float>(eosin_count, 0)				= Density.at<float>(row, col);  ++eosin_count;		break;
					case 3: background_density.at<float>(background_count, 0)	= Density.at<float>(row, col);  ++background_count; break;
				}
			}
		}

		ClassDensityRanges class_density_ranges;
		cv::meanStdDev(hema_density, class_density_ranges.hema_density_mean, class_density_ranges.hema_density_standard_deviation);
		cv::meanStdDev(eosin_density, class_density_ranges.eosin_density_mean, class_density_ranges.eosin_density_standard_deviation);
		cv::meanStdDev(background_density, class_density_ranges.background_density_mean, class_density_ranges.background_density_standard_deviation);
		return class_density_ranges;
	}

	std::pair<float, float> GetPercentile(float cx_cy_percentile, cv::Mat& cx_cy)
	{
		return GetPercentile(cx_cy_percentile, cx_cy_percentile, cx_cy);
	}

	std::pair<float, float> GetPercentile(float cx_percentile, float cy_percentile, cv::Mat& cx_cy)
	{
		cv::Mat sorted_cx_cy;
		cx_cy.copyTo(sorted_cx_cy);
		cv::sortIdx(sorted_cx_cy, sorted_cx_cy, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

		return std::pair<float, float>(cx_cy.at<float>(sorted_cx_cy.at<int>(sorted_cx_cy.rows / cx_percentile, 0), 0),
									   cx_cy.at<float>(sorted_cx_cy.at<int>(sorted_cx_cy.rows / cy_percentile, 1), 1));
	}

	MatrixRotationParameters RotateCxCy(cv::Mat& cx_cy, cv::Mat& output_matrix, cv::Mat& class_cx_cy)
	{
		if (cx_cy.data != output_matrix.data)
		{
			cx_cy.copyTo(output_matrix);
		}

		std::pair<float, float> class_median_cx_cy(GetPercentile(2, class_cx_cy));
		output_matrix.col(0) -= class_median_cx_cy.first;
		output_matrix.col(1) -= class_median_cx_cy.second;

		float angle = CovarianceCalculation(class_cx_cy);
		float theta = M_PI - angle;
		cv::Mat rotation_matrix((cv::Mat_<float>(2, 2) << cos(theta), sin(theta), -sin(theta), cos(theta)));

		for (size_t row = 0; row < output_matrix.rows; ++row)
		{
			output_matrix.at<float>(row, 0) = rotation_matrix.at<float>(0, 0)	* output_matrix.at<float>(row, 0) + rotation_matrix.at<float>(1, 0) * output_matrix.at<float>(row, 1);
			output_matrix.at<float>(row, 1) = rotation_matrix.at<float>(0, 1)	* output_matrix.at<float>(row, 0) + rotation_matrix.at<float>(1, 1) * output_matrix.at<float>(row, 1);
		}

		return { angle,	class_median_cx_cy.first, class_median_cx_cy.second };
	}

	MatrixRotationParameters RotateCxCy(cv::Mat& cx_cy, cv::Mat& output_matrix, float cx_median, float cy_median, float angle)
	{
		if (cx_cy.data != output_matrix.data)
		{
			cx_cy.copyTo(output_matrix);
		}

		output_matrix.col(0) -= cy_median;
		output_matrix.col(1) -= cy_median;

		float theta = M_PI - angle;
		cv::Mat rotation_matrix((cv::Mat_<float>(2, 2) << cos(theta), sin(theta), -sin(theta), cos(theta)));

		for (size_t row = 0; row < output_matrix.rows; ++row)
		{
			output_matrix.at<float>(row, 0) = rotation_matrix.at<float>(0, 0)	* output_matrix.at<float>(row, 0) + rotation_matrix.at<float>(1, 0) * output_matrix.at<float>(row, 1);
			output_matrix.at<float>(row, 1) = rotation_matrix.at<float>(0, 1)	* output_matrix.at<float>(row, 0) + rotation_matrix.at<float>(1, 1) * output_matrix.at<float>(row, 1);
		}

		return { angle,	cx_median, cy_median };
	}

	void RotateCxCyBack(cv::Mat& cx_cy_input, cv::Mat& output_matrix, float angle)
	{
		cv::Mat rotation_matrix(2, 2, CV_32FC1);
		rotation_matrix = (cv::Mat_<float>(2, 2) << cos(angle), sin(angle), -sin(angle), cos(angle));

		if (cx_cy_input.data != output_matrix.data)
		{
			cx_cy_input.copyTo(output_matrix);
		}

		for (size_t row = 0; row < output_matrix.rows; ++row)
		{
			output_matrix.at<float>(row, 0) = rotation_matrix.at<float>(0, 0)	* output_matrix.at<float>(row, 0) + rotation_matrix.at<float>(1, 0) * output_matrix.at<float>(row, 1);
			output_matrix.at<float>(row, 1) = rotation_matrix.at<float>(0, 1)	* output_matrix.at<float>(row, 0) + rotation_matrix.at<float>(1, 1) * output_matrix.at<float>(row, 1);
		}
	}

	void ScaleCxCy(const cv::Mat& rotated_input, cv::Mat& scaled_output, cv::Mat class_scale_params, cv::Mat& lut_scale_params)
	{
		ASAP::MiscMatrixOperations::PrepareOutputForInput(rotated_input, scaled_output);

		for (size_t row = 0; row < rotated_input.rows; ++row)
		{
			for (size_t col = 0; col < rotated_input.cols; ++col)
			{
				if (class_scale_params.at<float>(0, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) < class_scale_params.at<float>(1, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(0, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(0, col)) * (lut_scale_params.at<float>(1, col) - lut_scale_params.at<float>(0, col)) / (class_scale_params.at<float>(1, col) - class_scale_params.at<float>(0, col));
				else if (class_scale_params.at<float>(1, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) < class_scale_params.at<float>(2, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(1, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(1, col)) * (lut_scale_params.at<float>(2, col) - lut_scale_params.at<float>(1, col)) / (class_scale_params.at<float>(2, col) - class_scale_params.at<float>(1, col));
				else if (class_scale_params.at<float>(2, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) < class_scale_params.at<float>(3, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(2, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(2, col)) * (lut_scale_params.at<float>(3, col) - lut_scale_params.at<float>(2, col)) / (class_scale_params.at<float>(3, col) - class_scale_params.at<float>(2, col));
				else if (class_scale_params.at<float>(3, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) < class_scale_params.at<float>(4, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(3, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(3, col)) * (lut_scale_params.at<float>(4, col) - lut_scale_params.at<float>(3, col)) / (class_scale_params.at<float>(4, col) - class_scale_params.at<float>(3, col));
				else if (class_scale_params.at<float>(4, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) < class_scale_params.at<float>(5, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(4, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(4, col)) * (lut_scale_params.at<float>(5, col) - lut_scale_params.at<float>(4, col)) / (class_scale_params.at<float>(5, col) - class_scale_params.at<float>(4, col));
				else if (class_scale_params.at<float>(5, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) <= class_scale_params.at<float>(6, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(5, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(5, col)) * (lut_scale_params.at<float>(6, col) - lut_scale_params.at<float>(5, col)) / (class_scale_params.at<float>(6, col) - class_scale_params.at<float>(5, col));
			}
		}
	}

	void ScaleCxCyLUT(const cv::Mat& rotated_input, cv::Mat& scaled_output, cv::Mat class_scale_params, cv::Mat& lut_scale_params)
	{
		ASAP::MiscMatrixOperations::PrepareOutputForInput(rotated_input, scaled_output);

		for (size_t row = 0; row < rotated_input.rows; ++row)
		{
			for (size_t col = 0; col < rotated_input.cols; ++col)
			{
				if (class_scale_params.at<float>(0, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) < class_scale_params.at<float>(2, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(0, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(0, col)) * (lut_scale_params.at<float>(2, col) - lut_scale_params.at<float>(0, col)) / (class_scale_params.at<float>(2, col)- class_scale_params.at<float>(0, col));
				else if (class_scale_params.at<float>(2, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) < class_scale_params.at<float>(3, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(2, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(2, col)) * (lut_scale_params.at<float>(3, col) - lut_scale_params.at<float>(2, col)) / (class_scale_params.at<float>(3, col) - class_scale_params.at<float>(2, col));
				else if (class_scale_params.at<float>(3, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) < class_scale_params.at<float>(4, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(3, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(3, col)) * (lut_scale_params.at<float>(4, col) - lut_scale_params.at<float>(3, col)) / (class_scale_params.at<float>(4, col) - class_scale_params.at<float>(3, col));
				else if (class_scale_params.at<float>(4, col) <= rotated_input.at<float>(row, col) && rotated_input.at<float>(row, col) <= class_scale_params.at<float>(6, col))
					scaled_output.at<float>(row, col) = lut_scale_params.at<float>(4, col) + (rotated_input.at<float>(row, col) - class_scale_params.at<float>(4, col)) * (lut_scale_params.at<float>(6, col) - lut_scale_params.at<float>(4, col)) / (class_scale_params.at<float>(6, col) - class_scale_params.at<float>(4, col));
			}
		}
	}

	void TranslateCxCyBack(cv::Mat& cx_cy_rotated_back_class, cv::Mat& output_matrix, std::vector<cv::Point>& indices, float transform_x_median, float transform_y_median)
	{
		cv::Mat class_specific_cx_cy(cv::Mat::zeros(indices.size(), 2, CV_32FC1));
		size_t counter = 0;
		for (cv::Point& index : indices)
		{
			class_specific_cx_cy.at<float>(counter, 0) = cx_cy_rotated_back_class.at<float>(index.y, 0);
			class_specific_cx_cy.at<float>(counter, 1) = cx_cy_rotated_back_class.at<float>(index.y, 1);
		}

		cv::Mat sorted_class_specific_cx_cy;
		cv::sortIdx(class_specific_cx_cy, sorted_class_specific_cx_cy, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

		float x_median = class_specific_cx_cy.at<float>(sorted_class_specific_cx_cy.at<int>(sorted_class_specific_cx_cy.rows * 0.50, 0), 0);
		float y_median = class_specific_cx_cy.at<float>(sorted_class_specific_cx_cy.at<int>(sorted_class_specific_cx_cy.rows * 0.50, 1), 1);

		if (cx_cy_rotated_back_class.data != output_matrix.data)
		{
			cx_cy_rotated_back_class.copyTo(output_matrix);
		}

		output_matrix.col(0) = cx_cy_rotated_back_class.col(0) - x_median + transform_x_median;
		output_matrix.col(1) = cx_cy_rotated_back_class.col(1) - y_median + transform_y_median;
	}
}