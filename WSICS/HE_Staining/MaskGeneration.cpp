#include "MaskGeneration.h"

#define _USE_MATH_DEFINES
#include <iostream>
#include <math.h>

#include "../Misc/MatrixOperations.h"
#include "../Misc/MT_Singleton.hpp"

namespace WSICS::HE_Staining::MaskGeneration
{
	void ApplyBlur(const cv::Mat& input_matrix, cv::Mat& output_matrix, const uint32_t sigma)
	{
		WSICS::Misc::MatrixOperations::PrepareOutputForInput(input_matrix, output_matrix);

		cv::blur(output_matrix, output_matrix, cv::Size(sigma, sigma));
	}

	void ApplyCannyEdge(const cv::Mat& input_matrix, cv::Mat& output_matrix, const uint32_t low_threshold, const uint32_t high_threshold)
	{
		if (input_matrix.data != output_matrix.data)
		{
			input_matrix.copyTo(output_matrix);
		}

		double minimum_value, maximum_value;
		cv::minMaxIdx(output_matrix, &minimum_value, &maximum_value);

		output_matrix = output_matrix/ maximum_value * 255;
		output_matrix.convertTo(output_matrix, CV_8UC1);
		cv::Canny(output_matrix, output_matrix, low_threshold, high_threshold, 3);
	}

	std::vector<HoughTransform::Ellipse> ApplyHoughTransform(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const HoughTransform::RandomizedHoughTransformParameters& transform_parameters)
	{
		// Prepares the Hough Transform algorithm and executes it.
		HoughTransform::RandomizedHoughTransform hough_transform_algorithm(transform_parameters);
		return hough_transform_algorithm.Execute(binary_matrix, output_matrix, WSICS::BLOB_Operations::EIGHT_CONNECTEDNESS);
	}

	std::vector<HoughTransform::Ellipse> DetectEllipses(const cv::Mat& matrix,
		const uint32_t blur_sigma,
		const uint32_t canny_low_threshold,
		const uint32_t canny_high_threshold,
		const HoughTransform::RandomizedHoughTransformParameters& transform_parameters)
	{
		cv::Mat temporary_matrix;
		matrix.copyTo(temporary_matrix);

		ApplyBlur(temporary_matrix, temporary_matrix, blur_sigma);
		ApplyCannyEdge(temporary_matrix, temporary_matrix, canny_low_threshold, canny_high_threshold);

		return ApplyHoughTransform(temporary_matrix, temporary_matrix, transform_parameters);
	}

	double AcquirePercentile(std::vector<float> mean_vector, const float index_percentage)
	{
		size_t mean_vector_index = (mean_vector.size() - 1) * index_percentage;
		std::nth_element(mean_vector.begin(), mean_vector.begin() + mean_vector_index, mean_vector.end());
		return mean_vector[mean_vector_index];
	}

	std::vector<std::vector<cv::Point>> FilterContours(
		std::vector<std::vector<cv::Point>> & contours, 
		const std::vector<float>& density_mean, 
		const std::vector<float>& red_mean, 
		const std::vector<float>& blue_mean,
		const double density_mean_threshold,
		const double hema_mean_threshold)
	{
		if (contours.size() != density_mean.size() || contours.size() != red_mean.size() || contours.size() != blue_mean.size())
		{
			throw std::invalid_argument("Passed vectors do not correspond to each others sizes.");
		}

		std::vector<std::vector<cv::Point>> blue_contours;

		std::vector<std::vector<cv::Point>> new_contours;
		for (size_t contour = 0; contour < contours.size(); ++contour)
		{
			// first term is to remove low density faint objects - second term is to remove faint nuceli with too much red - third term is to remove blood cells
			if (density_mean[contour] < density_mean_threshold || red_mean[contour] < hema_mean_threshold)// || 1.5 * red_mean[contour] < blue_mean[contour]) // was 2 * bluemean[i] for LN 
			{
				if (1.5 * red_mean[contour] < blue_mean[contour])
				{
					blue_contours.push_back(contours[contour]);
				}

				new_contours.push_back(std::move(contours[contour]));
			}
		}		
		return new_contours;
	}

	EosinMaskInformation GenerateEosinMasks(const HSD::HSD_Model& hsd_image, const cv::Mat& background_mask, const HematoxylinMaskInformation& hema_mask_info, const float eosin_index_percentile)
	{
		// Merges the hema and background masks and generates the first eosin candidate mask.
		cv::Mat hema_and_background_mask(255 * (background_mask + hema_mask_info.full_mask));
		cv::Mat eosin_mask_candidate_one((255 - hema_and_background_mask) / 255);

		size_t candidates_count = cv::countNonZero(eosin_mask_candidate_one);
		std::vector<float> eosin_density_values;
		for (int row = 0; row < eosin_mask_candidate_one.rows; ++row)
		{
			for (int col = 0; col < eosin_mask_candidate_one.cols; ++col)
			{
				if (eosin_mask_candidate_one.at<unsigned char>(row, col) == 1)
				{
					eosin_density_values.emplace_back(hsd_image.red_density.at<float>(row, col));
				}
			}
		}

		// Calculates the eosin percentile.
		double eosin_percentile_value = AcquirePercentile(eosin_density_values, eosin_index_percentile);

		// Generates the seconds eosin candidate mask.
		cv::Mat eosin_mask_candidate_two(cv::Mat::zeros(hsd_image.red_density.size(), CV_8UC1));
		cv::threshold(hsd_image.red_density, eosin_mask_candidate_two, eosin_percentile_value, 1, 1);
		eosin_mask_candidate_two.convertTo(eosin_mask_candidate_two, CV_8UC1);
		eosin_mask_candidate_two = eosin_mask_candidate_two.mul(1 - background_mask);

		// Composes the eosin mask out of the two candidates and acquires the non-zero pixels.
		EosinMaskInformation eosin_mask_info;
		eosin_mask_info.full_mask = eosin_mask_candidate_one.mul(eosin_mask_candidate_two);
		std::vector<cv::Point> eosin_non_zero_pixels;
		cv::findNonZero(eosin_mask_info.full_mask, eosin_non_zero_pixels);
		std::shuffle(eosin_non_zero_pixels.begin(), eosin_non_zero_pixels.end(), WSICS::Misc::MT_Singleton::GetGenerator());

		// Creates a training mask, where each non-zero eosin mask pixel is set to 1 if there are more eosin pixels than hema pixels.
		size_t red_mask_pixels_sum = cv::sum(hema_mask_info.training_mask)[0];
		if (cv::sum(eosin_mask_info.full_mask)[0] > cv::sum(hema_mask_info.training_mask)[0])
		{
			eosin_mask_info.training_mask = cv::Mat::zeros(hsd_image.red_density.size(), CV_8UC1);
			for (size_t pixel = 0; pixel < red_mask_pixels_sum; ++pixel)
			{
				eosin_mask_info.training_mask.at<uchar>(eosin_non_zero_pixels[pixel]) = 1;
			}

			eosin_mask_info.training_pixels = red_mask_pixels_sum;
		}
		// If there are fewer eosin pixels compared to the hema pixels, set the eosin mask as the training mask.
		else
		{
			eosin_mask_info.training_mask	= eosin_mask_info.full_mask;
			eosin_mask_info.training_pixels = eosin_non_zero_pixels.size();
		}

		return eosin_mask_info;
	}

	std::pair<bool, HematoxylinMaskInformation> GenerateHematoxylinMasks(
		const HSD::HSD_Model& hsd_image,
		const cv::Mat& background_mask,
		const std::vector<HoughTransform::Ellipse>& ellipses,
		const float hema_index_percentile)
	{
		cv::Mat green_mask;
		cv::threshold(hsd_image.green_density, green_mask, 1.0, 1, cv::THRESH_BINARY);
		green_mask.convertTo(green_mask, CV_8UC1);

		cv::Mat difference_red_green(hsd_image.red_density - hsd_image.green_density);
		cv::threshold(difference_red_green, difference_red_green, 0.5, 1, cv::THRESH_BINARY);
		difference_red_green.convertTo(difference_red_green, CV_8UC1);
		difference_red_green = difference_red_green.mul(green_mask);

		HematoxylinMaskInformation hema_mask_info;

		if (cv::sum(difference_red_green)[0] < 10000) // Used to remove artifacts. - Good number = 10000
		{
			uint16_t interval = 200;
			std::vector<double> phi = LinearSpace(0, 2 * M_PI, interval);

			// Initializes the vectors that will hold the mean values, reservering enough space to equal the amount of ellipses.
			std::vector<float> red_mean;
			std::vector<float> blue_mean;
			std::vector<float> density_mean;
			red_mean.reserve(ellipses.size());
			blue_mean.reserve(ellipses.size());
			density_mean.reserve(ellipses.size());

			std::vector<std::vector<cv::Point>> contours;
			contours.reserve(ellipses.size());

			hema_mask_info.full_mask = cv::Mat::zeros(hsd_image.red_density.size(), CV_8UC1);

			// Loops through all the ellipses, calculating contour and mean values while updating the non-rejection training mask.
			for (const HoughTransform::Ellipse& ellipse : ellipses)
			{
				std::vector<std::vector<cv::Point>> coordinates(1);
				coordinates.reserve(interval);

				cv::Mat rotation_matrix = (cv::Mat_<double>(2, 2) << cos(ellipse.theta), sin(ellipse.theta), -sin(ellipse.theta), cos(ellipse.theta));
				for (uint16_t i = 0; i < interval; ++i)
				{
					coordinates[0].push_back
					(
						cv::Point
						(
							(rotation_matrix.at<double>(0, 0) * ellipse.major_axis *  cos(phi[i]) + rotation_matrix.at<double>(0, 1) * ellipse.minor_axis * sin(phi[i])) + ellipse.center.x,
							(rotation_matrix.at<double>(1, 0) * ellipse.major_axis *  cos(phi[i]) + rotation_matrix.at<double>(1, 1) * ellipse.minor_axis * sin(phi[i])) + ellipse.center.y
						)
					);
				}

				cv::Mat all_ellipse_raw(cv::Mat::zeros(hsd_image.red_density.size(), CV_8UC1));
				cv::drawContours(all_ellipse_raw, coordinates, 0, 255, cv::FILLED, 8);
				hema_mask_info.full_mask += all_ellipse_raw;

				contours.push_back(std::move(coordinates[0]));
				red_mean.push_back(cv::mean(hsd_image.red_density, all_ellipse_raw).val[0]);
				blue_mean.push_back(cv::mean(hsd_image.blue_density, all_ellipse_raw).val[0]);
				density_mean.push_back(cv::mean(hsd_image.density, all_ellipse_raw).val[0]);
			}

			// Initializes the training mask and removes artifacts from it.
			hema_mask_info.full_mask		/= 255;
			hema_mask_info.training_mask	= hema_mask_info.full_mask.clone();

			// Acquires the hema and density percentiles.
			double hema_mean_threshold		= AcquirePercentile(red_mean, hema_index_percentile);
			double density_mean_threshold	= AcquirePercentile(density_mean, 0.02f);

			contours = FilterContours(contours, density_mean, red_mean, blue_mean, density_mean_threshold, hema_mean_threshold);
			cv::drawContours(hema_mask_info.training_mask, contours, -1, 255, cv::FILLED, 8);

			hema_mask_info.training_mask = hema_mask_info.training_mask - hema_mask_info.training_mask.mul(background_mask);
			hema_mask_info.training_pixels = 0;

			for (int row = 0; row < hema_mask_info.training_mask.rows; ++row)
			{
				for (int column = 0; column < hema_mask_info.training_mask.cols; ++column)
				{
					if (hema_mask_info.training_mask.at<unsigned char>(row, column) == 1)
					{
						if (hsd_image.red_density.at<float>(row, column) < hema_mean_threshold)
						{
							hema_mask_info.training_mask.at<unsigned char>(row, column) = 0;
						}
						else
						{
							++hema_mask_info.training_pixels;
						}
					}
				}
			}
		}
		else
		{
			return std::pair<bool, HematoxylinMaskInformation>(false, hema_mask_info);
		}

		return std::pair<bool, HematoxylinMaskInformation>(true, hema_mask_info);
	}

	std::vector<double> LinearSpace(const double start_point, const double end_point, const uint16_t interval)
	{
		std::vector<double> linear_vector;
		linear_vector.reserve(interval);
		for (size_t i = 0; i < interval; ++i)
		{
			linear_vector.push_back(start_point + i * (end_point - start_point) / (interval - 1));
		}
		return linear_vector;
	}
}