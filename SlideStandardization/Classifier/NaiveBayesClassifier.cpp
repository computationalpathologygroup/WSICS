#include "NaiveBayesClassifier.h"

#include <numeric>
#include <float.h>
#include <math.h>
#include <set>

namespace ML
{
	NaiveBayesClassifier::NaiveBayesClassifier(const size_t bins, const float blur, const ProbabilityIntegration integration_type)
		: n_bins(bins), blur_sigma(blur), probability_integration_type(integration_type)
	{
	}

	NaiveBayesClassifier::NaiveBayesClassifier(const NaiveBayesClassifier& other)
		: Classifier(other), n_bins(other.n_bins), blur_sigma(other.blur_sigma),
		probability_integration_type(other.probability_integration_type)
	{
	}

	void NaiveBayesClassifier::TrainClassifier$(const cv::ml::TrainData& train_data)
	{
		// Verify that parameters are correct.
		if (n_bins == 0 || blur_sigma <= 0)
		{
			throw std::runtime_error("The number of bins must be at least 1 or higher. The sigma must be above 0.");
		}

		cv::Mat samples = train_data.getSamples();
		cv::Mat responses = train_data.getResponses();
		{
			std::set<int32_t> classes;
			for (size_t response = 0; response < responses.rows; ++response)
			{
				classes.insert((int32_t)responses.at<float>(response, 0));
			}

			m_classes$.clear();
			for (const int32_t class_label : classes)
			{
				m_classes$.push_back(class_label);
			}
		}

		m_feature_classifiers_.clear();
		std::vector<float> feature_samples(train_data.getNSamples()); // input features
		std::vector<int32_t> feature_responses(train_data.getNSamples()); // output labels
		for (size_t feature = 0; feature < samples.cols; ++feature)
		{
			// Construct vectors with the input (for this feature) and the output.
			for (size_t sample = 0; sample < samples.rows; ++sample)
			{
				feature_samples[sample] = samples.at<float>(sample, feature);
				feature_responses[sample] = static_cast<int32_t>(responses.at<float>(sample, 0));
			}

			NaiveBayesFeatureClassifier feature_classifier;
			feature_classifier.Train(feature_samples, feature_responses, GetNrOfClasses(), n_bins, blur_sigma);
			m_feature_classifiers_.push_back(feature_classifier);
		}

		// Set the classifier as trained.
		m_is_trained$ = true;
	}

	void NaiveBayesClassifier::ClassifySamples$(const cv::Mat& input, cv::Mat& output) const
	{
		// Search for the maximum output and set the class label accordingly.
		cv::Mat posterior_output;
		PosteriorSamples$(input, posterior_output);

		output = cv::Mat::zeros(input.rows, 1, CV_8UC1);
		for (size_t row = 0; row < posterior_output.rows; ++row)
		{
			double min, max;
			cv::minMaxLoc(output.row(row), &min, &max);
			output.at<uchar>(row, 0) = m_classes$[static_cast<uchar>(max)];
		}
	}

	void NaiveBayesClassifier::PosteriorSamples$(const cv::Mat& input, cv::Mat& output) const
	{
		size_t features = input.cols;
		if (features < 1 || features != m_feature_classifiers_.size())
		{
			throw std::runtime_error("Amount of features don't align with the trained features.");
		}

		// Aggregate the probabilities for all features
		output = cv::Mat::zeros(input.rows, this->GetNrOfClasses(), CV_32FC1);
		output += 1.0;

		cv::Mat feature_output;
		for (size_t feature = 0; feature < features; ++feature)
		{
			for (size_t sample = 0; sample < input.rows; ++sample)
			{
			//	std::cout << "New input " << std::setprecision(40) << input.at<float>(sample, feature) << std::endl;
				m_feature_classifiers_[feature].Run(input.at<float>(sample, feature), feature_output);
				if (this->probability_integration_type == ProbabilityIntegration::ADDITION)
				{
					output.row(sample) += feature_output;
				}
				else
				{
					output.row(sample) = output.row(sample).mul(feature_output);
				}
			}
		}

		cv::Mat norms;
		cv::reduce(output, norms, 1, CV_REDUCE_SUM, CV_64F);
		for (size_t row = 0; row < output.rows; ++row)
		{
			std::vector<float> test;

			for (size_t feature = 0; feature < features; ++feature)
			{
				//std::cout << "New value " << std::setprecision(40) << output.at<float>(row, feature) << std::endl;

				//test.push_back(output.at<float>(row, feature));
				//norm += output.at<float>(row, feature);
			}

			long double norm = std::accumulate(test.begin(), test.end(), (long double)(0));
			norm = norms.at<long double>(row, 0);
			//std::cout << "New norm " << std::setprecision(40) << norm << std::endl;

			if (norm == 0)
			{
				output.row(row).setTo(cv::Scalar(static_cast<float>(1) / static_cast<float>(this->GetNrOfClasses())));
			}
			else
			{
				for (size_t feature = 0; feature < features; ++feature)
				{
					output.at<float>(row, feature) /= norm;
				}
			}
		}
	}
}