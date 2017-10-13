#include "HSD_Model.h"

#include <vector>

namespace HSD
{
	HSD_Model::HSD_Model(const cv::Mat& rgb_image, const HSD_Initialization_Type initialization_type) : m_initialization_type_(initialization_type)
	{
		std::vector<cv::Mat> channels(3);
		cv::split(rgb_image, channels);

		if (initialization_type == HSD_Initialization_Type::STANDARD)
		{
			this->red_density = std::move(channels[0]);
			this->green_density = std::move(channels[1]);
			this->blue_density = std::move(channels[2]);
		}
		else
		{
			this->red_density = std::move(channels[2]);
			this->green_density = std::move(channels[1]);
			this->blue_density = std::move(channels[0]);
		}

		AdjustChannelMatrix_(this->red_density);
		AdjustChannelMatrix_(this->green_density);
		AdjustChannelMatrix_(this->blue_density);

		//convertScaleAbs(Blue_density,Blue_density);
		this->density = (this->red_density + this->green_density + this->blue_density) / 3;
		this->c_x = this->red_density / this->density - 1;
		this->c_y = (this->green_density - this->red_density) / (sqrt(3.) * this->density);
	}

	HSD_Initialization_Type HSD_Model::GetInitializationType(void)
	{
		return m_initialization_type_;
	}

	cv::Mat HSD_Model::GetRGB(void)
	{
		std::vector<cv::Mat> color_channels(3);
		for (cv::Mat& channel : color_channels)
		{
			channel = cv::Mat::zeros(c_x.size(), CV_32FC1);
		}

		cv::Mat d_red = 0.2 *(c_x + 1);
		cv::Mat d_green = 0.5 *0.2*(2 - c_x + sqrt(3.) * c_y);
		cv::Mat d_blue = 0.5 * 0.2*(2 - c_x - sqrt(3.) * c_y);
		for (int i = 0; i < c_x.rows; ++i)
		{
			for (int j = 0; j < c_x.cols; ++j)
			{
				color_channels[0].at<float>(i, j) = 255 * exp(-d_red.at<float>(i, j));
				color_channels[1].at<float>(i, j) = 255 * exp(-d_green.at<float>(i, j));
				color_channels[2].at<float>(i, j) = 255 * exp(-d_blue.at<float>(i, j));
			}
		}

		cv::Mat normalized_image;
		cv::merge(color_channels, normalized_image);
		normalized_image.convertTo(normalized_image, CV_8UC3);

		return normalized_image;
	}

	void HSD_Model::AdjustChannelMatrix_(cv::Mat& matrix)
	{
		for (size_t row = 0; row < matrix.rows; ++row)
		{
			uchar* row_ptr = matrix.ptr(row);
			for (size_t col = 0; col < matrix.cols; ++col)
			{
				if (*row_ptr == 0)
				{
					*row_ptr = 1;

				}
				else if (*row_ptr == 255)
				{
					*row_ptr = 254;
				}
				++row_ptr;
			}
		}

		matrix.convertTo(matrix, CV_32F);
		matrix /= 255;
		cv::log(matrix, matrix);
		matrix *= -1;
	}
}