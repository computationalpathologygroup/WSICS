#ifndef __HSDmodel_H
#define __HSDmodel_H

#include "opencv2/core.hpp"

class HSDmodel{

public:

	~HSDmodel();
	// defining data members for HSD color model
	cv::Mat Red_density, Green_density, Blue_density, Density, Cx, Cy;
	cv::Mat NormalizedImage;

	// Calculating HSD color model
	void calculateHSD(cv::Mat& image);
	void calculateHSD2(cv::Mat& image);

	// Calculating HSD reverse transform to get bak to RGB color model
	void HSDreverse(cv::Mat& Scaled_DensityIamge, cv::Mat& Cx_imgNorm, cv::Mat& Cy_imgNorm);
	void HSDreverse(cv::Mat& Cx_imgNorm, cv::Mat& Cy_imgNorm);
};
#endif