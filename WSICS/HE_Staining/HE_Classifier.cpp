#include "HE_Classifier.h"

namespace WSICS::HE_Staining
{
	HE_Classifier::HE_Classifier(uint32_t max_leaf_size, uint32_t k_value) : max_leaf_size(max_leaf_size), k_value(k_value)
	{
	}

	ClassificationResults HE_Classifier::Classify(HSD::HSD_Model& hsd_image, cv::Mat& background_mask, HematoxylinMaskInformation& hema_mask_info, EosinMaskInformation& eosin_mask_info)
	{
		// Acquires the mask information.
		ClassificationResults results;
		results.train_and_class_data = CreateTrainAndClassData_(hsd_image, background_mask, hema_mask_info, eosin_mask_info);

		// Acquires the classification information.
		std::pair<cv::Mat, cv::Mat> class_info(this->Apply_KNN_(hsd_image, background_mask, hema_mask_info, eosin_mask_info, results.train_and_class_data));
		results.tissue_classes	= std::move(class_info.first);
		results.all_classes		= std::move(class_info.second);

		// Counts the amount of pixels per class.
		results.hema_pixels			= 0;
		results.eosin_pixels		= 0;
		results.background_pixels	= 0;
		for (int row = 0; row < results.all_classes.rows; ++row)
		{
			for (int col = 0; col < results.all_classes.cols; ++col)
			{
				switch (results.all_classes.at<uchar>(row, col))
				{
					case 1: ++results.hema_pixels;			break;
					case 2: ++results.eosin_pixels;			break;
					case 3: ++results.background_pixels;	break;
				}
			}
		}

		return results;
	}

	std::pair<cv::Mat, cv::Mat> HE_Classifier::Apply_KNN_(
		const HSD::HSD_Model& hsd_image,
		const cv::Mat& background_mask,
		const HematoxylinMaskInformation& hema_mask_info,
		const EosinMaskInformation& eosin_mask_info,
		const TrainAndClassData& train_and_class_data)
	{
		const cv::Mat&					class_data(train_and_class_data.class_data);
		const cv::Mat&					train_data(train_and_class_data.train_data);
		const cv::Mat&					test_data(train_and_class_data.test_data);
		const std::vector<cv::Point>&	test_indices(train_and_class_data.test_indices);

		// Copies the train and test data into flann matrices.
		cvflann::Matrix<float> flann_train_data(new float[train_data.rows * train_data.cols], train_data.rows, train_data.cols);
		for (uint32_t x = 0; x < train_data.rows; ++x)
		{
			for (uint32_t y = 0; y < train_data.cols; ++y)
			{
				flann_train_data[x][y] = train_data.at<float>(x, y);
			}
		}

		cvflann::Matrix<float> flann_test_data(new float[test_data.rows * test_data.cols], test_data.rows, test_data.cols);
		for (uint32_t x = 0; x < test_data.rows; ++x)
		{
			for (uint32_t y = 0; y < test_data.cols; ++y)
			{
				flann_test_data[x][y] = test_data.at<float>(x, y);
			}
		}

		// Initializes the knn tree.
		cvflann::KDTreeSingleIndexParams index_parameters(this->max_leaf_size);
		cvflann::KDTreeSingleIndex<cvflann::L2<float>> tree_model(flann_train_data, index_parameters);
		tree_model.buildIndex();

		// Creates two flann matrices to hold the results and executes the K-NN search.
		cvflann::Matrix<int> flann_indices(new int[this->k_value * test_data.rows], test_data.rows, this->k_value);
		cvflann::Matrix<float> flann_distributions(new float[this->k_value * test_data.rows], test_data.rows, this->k_value);
		tree_model.knnSearch(flann_test_data, flann_indices, flann_distributions, this->k_value, cvflann::SearchParams(128));

		std::vector<float> value_list(this->k_value);
		cv::Mat predicted(cv::Mat::zeros(test_data.rows, 1, CV_32FC1));
		for (uint32_t x = 0; x < flann_indices.rows; ++x)
		{
			for (uint32_t y = 0; y < this->k_value; ++y)
			{
				value_list[y] = class_data.at<float>(flann_indices[x][y], 0);
			}

			if (std::accumulate(value_list.begin(), value_list.end(), 0) > 0)
			{
				predicted.at<float>(x, 0) = 1;
			}
			else
			{
				predicted.at<float>(x, 0) = -1;
			}
		}

		// Generates the tissue_class matrix, based on the results of the prediction matrix.
		predicted.convertTo(predicted, CV_32FC1);
		cv::Mat tissue_classes = cv::Mat::zeros(hsd_image.density.rows, hsd_image.density.cols, CV_32FC1);
		for (size_t index = 0; index < test_indices.size(); ++index)
		{
			if (predicted.at<float>(index, 0) == 1)
			{
				tissue_classes.at<float>(test_indices[index]) = 1;
			}
			else
			{
				tissue_classes.at<float>(test_indices[index]) = 2;
			}
		}

		// Converts the tissue_class matrix and prepares the all_tissue_classes matrix.
		tissue_classes.convertTo(tissue_classes, CV_8UC1);
		cv::Mat all_classes(tissue_classes + hema_mask_info.full_mask + 2 * eosin_mask_info.training_mask + 3 * background_mask);
		cv::medianBlur(all_classes, all_classes, 3);

		return { tissue_classes, all_classes };
	}

	cv::Mat HE_Classifier::CalculateOneStdDevBelowMean_(cv::Mat& matrix)
	{
		cv::Mat mean, standard_deviation;
		cv::meanStdDev(matrix, mean, standard_deviation);
		return (matrix - mean) / standard_deviation;
	}

	TrainAndClassData HE_Classifier::CreateTrainAndClassData_(
		HSD::HSD_Model& hsd_image,
		cv::Mat& background_mask,
		HematoxylinMaskInformation& hema_mask_info,
		EosinMaskInformation& eosin_mask_info)
	{
		cv::Mat c_x_new(CalculateOneStdDevBelowMean_(hsd_image.c_x));
		cv::Mat c_y_new(CalculateOneStdDevBelowMean_(hsd_image.c_y));
		cv::Mat density_new(CalculateOneStdDevBelowMean_(hsd_image.density));

		// Acquires the non-zero eosin indices. If larger than 0, 
		std::vector<cv::Point> eosin_indices;
		cv::findNonZero(eosin_mask_info.training_mask, eosin_indices);
		if (eosin_indices.size() > 0)
		{
			// Generates the training data.
			size_t train_counter = 0;

			std::vector<cv::Point> hema_indices;
			cv::findNonZero(hema_mask_info.training_mask, hema_indices);
			TrainAndClassData train_and_class_data;

			train_and_class_data.train_data = cv::Mat::zeros(hema_indices.size() + eosin_indices.size(), 3, CV_32FC1);
			train_and_class_data.class_data = cv::Mat::zeros(hema_indices.size() + eosin_indices.size(), 1, CV_32FC1);
			for (const cv::Point2f& hema_point : hema_indices)
			{
				train_and_class_data.train_data.at<float>(train_counter, 0) = c_x_new.at<float>(hema_point);
				train_and_class_data.train_data.at<float>(train_counter, 1) = c_y_new.at<float>(hema_point);
				train_and_class_data.train_data.at<float>(train_counter, 2) = density_new.at<float>(hema_point);
				train_and_class_data.class_data.at<float>(train_counter, 0) = 1;
				++train_counter;
			}

			for (const cv::Point2f& eosin_point : eosin_indices)
			{
				train_and_class_data.train_data.at<float>(train_counter, 0) = c_x_new.at<float>(eosin_point);
				train_and_class_data.train_data.at<float>(train_counter, 1) = c_y_new.at<float>(eosin_point);
				train_and_class_data.train_data.at<float>(train_counter, 2) = density_new.at<float>(eosin_point);
				train_and_class_data.class_data.at<float>(train_counter, 0) = -1;
				++train_counter;
			}

			// Generates the test data.
			size_t test_counter = 0;

			cv::Mat test_matrix = cv::Mat::ones(hsd_image.red_density.size(), CV_8UC1) - (eosin_mask_info.training_mask + hema_mask_info.training_mask + background_mask);
			cv::findNonZero(test_matrix, train_and_class_data.test_indices);
			train_and_class_data.test_data = cv::Mat::zeros(train_and_class_data.test_indices.size(), 3, CV_32FC1);
			for (const cv::Point2f& test_point : train_and_class_data.test_indices)
			{
				train_and_class_data.test_data.at<float>(test_counter, 0) = c_x_new.at<float>(test_point);
				train_and_class_data.test_data.at<float>(test_counter, 1) = c_y_new.at<float>(test_point);
				train_and_class_data.test_data.at<float>(test_counter, 2) = density_new.at<float>(test_point);
				++test_counter;
			}

			return train_and_class_data;
		}

		return TrainAndClassData();
	}
}