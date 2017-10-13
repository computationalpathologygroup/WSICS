#ifndef __HSD_TRANSFORMATIONS_H__
#define __HSD_TRANSFORMATIONS_H__

#include "opencv2/core.hpp"

namespace HSD
{
	void CxCyToRGB(cv::Mat& cx_cy_input, cv::Mat& output_matrix);
	void CxCyToRGB(cv::Mat& cx_cy_input, cv::Mat& output_matrix, cv::Mat& density_scaling);
}
#endif // __HSD_TRANSFORMATIONS_H__