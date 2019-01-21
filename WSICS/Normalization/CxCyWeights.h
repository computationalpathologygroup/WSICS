#ifndef __WSICS_NORMALIZATION_CXYWEIGHTS__
#define __WSICS_NORMALIZATION_CXYWEIGHTS__

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include "../ML/NaiveBayesClassifier.h"

/// <Summary>
/// This namespace is for refining the rigid transformation of CxCy values
/// The weights are applied based on the amount of stain for each pixel
/// </summary>
namespace WSICS::Normalization::CxCyWeights
{
	struct Weights
	{
		cv::Mat hema;
		cv::Mat eosin;
		cv::Mat background;
	};

	// applies the already calculated weights on the transformes CxCy values
	cv::Mat ApplyWeights(const cv::Mat& c_xy_hema, const cv::Mat& c_xy_eosin, const cv::Mat& c_xy_background, const Weights& weights);

	/// <summary>
	/// Creates and trains a NaiveBayesClassifier.
	/// </summary>
	ML::NaiveBayesClassifier CreateNaiveBayesClassifier(const cv::Mat& c_x, const cv::Mat& c_y, const cv::Mat& density, const cv::Mat& all_tissue_classes);
};
#endif // __WSICS_NORMALIZATION_CXYWEIGHTS__