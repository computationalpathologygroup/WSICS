//===========================================================================
// This code classifies all the pixels in an image into Hema, Eos, and Background
// The training samples are initially prepared by hough transfrom and then automatic
// percentile thresholding.
//===========================================================================

#ifndef __ClassifyHE_H
#define __ClassifyHE_H

#include <opencv2/core/core.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/flann/flann_base.hpp>
#include <opencv2/flann/kdtree_single_index.h>
#include <numeric>

#include "../HSD/HSD_Model.h"
#include "../HE_Staining/MaskGeneration.h"

namespace HE_Staining
{
	/// <summary>
	/// 
	/// </summary>
	struct TrainAndClassData
	{
		cv::Mat class_data;
		cv::Mat train_data;
		cv::Mat test_data;
		std::vector<cv::Point> test_indices;
	};

	/// <summary>
	/// 
	/// </summary>
	struct ClassificationResults
	{
		cv::Mat all_classes;
		cv::Mat tissue_classes;
		size_t hema_pixels;
		size_t eosin_pixels;
		size_t background_pixels;
		TrainAndClassData train_and_class_data;
	};



	/// <summary>
	/// This class can aid in the classification of HE stained tissue images. It does
	/// this through the application of a trained K-NN algorithm with Kd-trees.
	/// </summary>
	class HE_Classifier
	{
		public:
			uint32_t max_leaf_size;
			uint32_t k_value;

			/// <summary>
			/// Initializes the classifier, setting the max leaf size and k value that direct the K-NN execution.
			/// </summary>
			/// <param name="max_leaf_size">The max leaf size for the tree, used by the K-NN algorithm.</param>
			/// <param name="k_value">The K value for the K-NN algorithm.</param>
			HE_Classifier(uint32_t max_leaf_size = 50, uint32_t k_value = 7);

			/// <summary>
			/// Performs the classification of the image, using the background, Eosin and Hematoxylin masks to
			/// train a K-NN tree, which in turn is used to provide the actual classification.
			/// </summary>
			/// <param name="hsd_image">The image to perform the classification on.</param>
			/// <param name="background_mask">A matrix annotating the background pixels.</param>
			/// <param name="hema_mask_info">The Hematoxylin masks, annotating pixels corresponding to the staining and those used for training.</param>
			/// <param name="eosin_mask_info">The Eosin masks, annotating pixels corresponding to the staining and those used for training.</param>
			/// <returns></returns>
			ClassificationResults Classify(HSD::HSD_Model& hsd_image, cv::Mat& background_mask, HematoxylinMaskInformation& hema_mask_info, EosinMaskInformation& eosin_mask_info);

		private:
			/// <summary>
			/// Classifies the image through the application of K-NN.
			/// </summary>
			/// <param name="hsd_image">The image to perform the classification on.</param>
			/// <param name="background_mask">A matrix annotating the background pixels.</param>
			/// <param name="hema_mask_info">The Hematoxylin masks, annotating pixels corresponding to the staining and those used for training</param>
			/// <param name="eosin_mask_info">The Eosin masks, annotating pixels corresponding to the staining and those used for training.</param>
			/// <param name="train_and_class_data">The generated struct with the specific class, training and test data.</param>
			/// <returns>A pair containing two matrices which together hold the classification information.</returns>
			std::pair<cv::Mat, cv::Mat> Apply_KNN_(
				const HSD::HSD_Model& hsd_image,
				const cv::Mat& background_mask,
				const HematoxylinMaskInformation& hema_mask_info,
				const EosinMaskInformation& eosin_mask_info,
				const TrainAndClassData& train_and_class_data);
			/// <summary>
			/// Calculates the mean and standard deviation values for the matrix. Subtracting the latter from the former.
			/// </summary>
			/// <param name="matrix">The matrix to perform the operation on.</param>
			/// <returns>A matrix containing the mean values, subtracted by the standard deviation values.</returns>
			cv::Mat CalculateOneStdDevBelowMean_(cv::Mat& matrix);
			/// <summary>
			/// Creates the training and test data for the K-NN tree.
			/// </summary>
			/// <param name="hsd_image">The image to perform the classification on.</param>
			/// <param name="background_mask">A matrix annotating the background pixels.</param>
			/// <param name="hema_mask_info">The Hematoxylin masks, annotating pixels corresponding to the staining and those used for training.</param>
			/// <param name="eosin_mask_info">The Eosin masks, annotating pixels corresponding to the staining and those used for training.</param>
			/// <returns>A struct containing the class, training and test data.</returns>
			TrainAndClassData CreateTrainAndClassData_(HSD::HSD_Model& hsd_image, cv::Mat& background_mask, HematoxylinMaskInformation& hema_mask_info, EosinMaskInformation& eosin_mask_info);
	};
}
#endif