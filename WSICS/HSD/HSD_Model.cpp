#include "HSD_Model.h"

#include <vector>

namespace WSICS::HSD
{
	HSD_Model::HSD_Model(void)
	{
	}

	HSD_Model::HSD_Model(const cv::Mat& rgb_image, const HSD_Initialization_Type initialization_type) : m_initialization_type_(initialization_type)
	{
		std::vector<cv::Mat> channels(3);
		cv::split(rgb_image, channels);

		if (initialization_type == HSD_Initialization_Type::RGB)
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

		this->density	= (this->red_density + this->green_density + this->blue_density) / 3;
		this->c_x		= this->red_density / density - 1;
		this->c_y		= (this->green_density - this->blue_density) / (sqrt(3.) * density);
	}

	HSD_Initialization_Type HSD_Model::GetInitializationType(void) const
	{
		return m_initialization_type_;
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