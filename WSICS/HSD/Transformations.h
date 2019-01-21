#ifndef __WSICS_HSD_TRANSFORMATIONS__
#define __WSICS_HSD_TRANSFORMATIONS__

#include "opencv2/core.hpp"

namespace WSICS::HSD
{
	void CxCyToRGB(const cv::Mat& cx_cy_input, cv::Mat& output_matrix);
	void CxCyToRGB(const cv::Mat& cx_cy_input, cv::Mat& output_matrix, const cv::Mat& density_scaling);
}
#endif // __WSICS_HSD_TRANSFORMATIONS__