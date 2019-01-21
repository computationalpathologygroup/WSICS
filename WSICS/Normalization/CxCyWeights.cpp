#include "CxCyWeights.h"

namespace WSICS::Normalization::CxCyWeights
{
	cv::Mat ApplyWeights(const cv::Mat& c_xy_hema, const cv::Mat& c_xy_eosin, const cv::Mat& c_xy_background, const Weights& weights)
	{
		cv::Mat c_xy_normalized(c_xy_hema.rows, 2, CV_32FC1);

		size_t counter = 0;
		for (size_t row = 0; row < weights.hema.rows; ++row)
		{
			for (size_t col = 0; col < weights.hema.cols; ++col)
			{
				c_xy_normalized.at<float>(counter, 0) = c_xy_hema.at<float>(counter, 0) * weights.hema.at<float>(row, col) + c_xy_eosin.at<float>(counter, 0) * weights.eosin.at<float>(row, col) + c_xy_background.at<float>(counter, 0) * weights.background.at<float>(row, col);
				c_xy_normalized.at<float>(counter, 1) = c_xy_hema.at<float>(counter, 1) * weights.hema.at<float>(row, col) + c_xy_eosin.at<float>(counter, 1) * weights.eosin.at<float>(row, col) + c_xy_background.at<float>(counter, 1) * weights.background.at<float>(row, col);
				++counter;
			}
		}

		return c_xy_normalized;
	}

	ML::NaiveBayesClassifier CreateNaiveBayesClassifier(const cv::Mat& c_x, const cv::Mat& c_y, const cv::Mat& density, const cv::Mat& class_data)
	{
		cv::Mat samples(class_data.rows, 3, CV_32FC1);
		cv::Mat responses(class_data.rows, 1, CV_32S);

		size_t current_sample = 0;
		for (size_t row = 0; row < class_data.rows; ++row)
		{
			samples.at<float>(current_sample, 0) = c_x.at<float>(row, 0);
			samples.at<float>(current_sample, 1) = c_y.at<float>(row, 0);
			samples.at<float>(current_sample, 2) = density.at<float>(row, 0) / 2;

			switch ((int32_t)class_data.at<float>(row, 0))
			{
				case 1: responses.at<float>(current_sample, 0) = 0; break;
				case 2: responses.at<float>(current_sample, 0) = 1; break;
				case 3: responses.at<float>(current_sample, 0) = 2; break;
			}
				
			++current_sample;
		}

		cv::Ptr<cv::ml::TrainData> train_data(cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE, responses));
		ML::NaiveBayesClassifier classifier;
		classifier.Train(*train_data);

		return classifier;
	}

	Weights GenerateWeights(const cv::Mat& c_x, const cv::Mat& c_y, const cv::Mat& density, const ML::NaiveBayesClassifier& classifier)
	{
		cv::Mat classifier_input(cv::Mat::zeros(c_x.rows, 3, CV_32FC1));
		for (size_t row = 0; row < c_x.rows; ++row)
		{
			classifier_input.at<float>(row, 0) = c_x.at<float>(row, 0);
			classifier_input.at<float>(row, 1) = c_y.at<float>(row, 0);
			classifier_input.at<float>(row, 2) = density.at<float>(row, 0) / 2;
		}

		cv::Mat posteriors;
		classifier.Posterior(classifier_input, posteriors);

		Weights weights { cv::Mat::zeros(c_x.rows, c_x.cols, CV_32FC1), cv::Mat::zeros(c_x.rows, c_x.cols, CV_32FC1), cv::Mat::zeros(c_x.rows, c_x.cols, CV_32FC1) };

		for (size_t row = 0; row < c_x.rows; ++row)
		{
			weights.hema.at<float>(row, 0)			= posteriors.at<float>(row, 0);
			weights.eosin.at<float>(row, 0)			= posteriors.at<float>(row, 1);
			weights.background.at<float>(row, 0)	= posteriors.at<float>(row, 2);
		}

		return weights;
	}
}