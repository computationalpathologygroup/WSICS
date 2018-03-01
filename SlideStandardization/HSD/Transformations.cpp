#include "Transformations.h"

#include <opencv2\highgui.hpp>

namespace HSD
{
	void CxCyToRGB(const cv::Mat& cx_cy_input, cv::Mat& output_matrix)
	{
		cv::Mat temporary_matrix;
		CxCyToRGB(cx_cy_input, output_matrix, temporary_matrix);
	}

	void CxCyToRGB(const cv::Mat& cx_cy_input, cv::Mat& output_matrix, const cv::Mat& density_scaling)
	{
		// Calculates the densities for each channel.
		cv::Mat channel_red;
		cv::Mat channel_green;
		cv::Mat channel_blue;

		if (density_scaling.empty())
		{
			channel_red		= 0.2 * (cx_cy_input.col(0) + 1);
			channel_green	= 0.5 * 0.2*(2 - cx_cy_input.col(0) + sqrt(3.) * cx_cy_input.col(1));
			channel_blue	= 0.5 * 0.2*(2 - cx_cy_input.col(0) - sqrt(3.) * cx_cy_input.col(1));
		}
		else
		{
			channel_red		= density_scaling.mul(cx_cy_input.col(0) + 1);
			channel_green	= 0.5 * density_scaling.mul(2 - cx_cy_input.col(0) + sqrt(3.) * cx_cy_input.col(1));
			channel_blue	= 0.5 * density_scaling.mul(2 - cx_cy_input.col(0) - sqrt(3.) * cx_cy_input.col(1));

		}

		// Converts the values for each channel.
		for (size_t row = 0; row < cx_cy_input.rows; ++row)
		{
			for (size_t col = 0; col < cx_cy_input.cols; ++col)
			{
				channel_red.at<float>(row, col)		= 255 * std::exp(-channel_red.at<float>(row, col));
				channel_green.at<float>(row, col)	= 255 * std::exp(-channel_green.at<float>(row, col));
				channel_blue.at<float>(row, col)	= 255 * std::exp(-channel_blue.at<float>(row, col));
			}
		}

		std::vector<cv::Mat> channels{ channel_blue, channel_green, channel_red };
		output_matrix.convertTo(output_matrix, CV_32FC3);
		cv::merge(channels, output_matrix);
		output_matrix.convertTo(output_matrix, CV_8UC3);
	}
}