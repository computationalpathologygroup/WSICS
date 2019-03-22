#include "NaiveBayesClassifier.h"

#include <numeric>
#include <float.h>
#include <math.h>
#include <set>

namespace WSICS::ML
{
	NaiveBayesClassifier::NaiveBayesClassifier(const uint64_t bins, const float blur, const ProbabilityIntegration integration_type)
		: n_bins(bins), blur_sigma(blur), probability_integration_type(integration_type), m_is_trained_(false)
	{
	}

	const std::vector<uchar> NaiveBayesClassifier::GetClasses(void) const
	{
		return m_classes_;
	}

	const std::vector<std::string>& NaiveBayesClassifier::GetFeatureNames(void) const
	{
		return m_trained_feature_names_;
	}
	const std::vector<double>& NaiveBayesClassifier::GetWeights(void) const
	{
		return m_weights_;
	}

	void NaiveBayesClassifier::SetWeights(const std::vector<double>& weights)
	{
		m_weights_ = weights;
	}

	bool NaiveBayesClassifier::IsTrained(void) const
	{
		return m_is_trained_;
	}

	void NaiveBayesClassifier::Classify(const cv::Mat& input, cv::Mat& output) const
	{
		// Ensures this function isn't called on an untrained classifier.
		CheckIfTrained_();

		// Search for the maximum output and set the class label accordingly.
		cv::Mat posterior_output;
		Posterior(input, posterior_output);

		output = cv::Mat::zeros(input.rows, 1, CV_8UC1);
		for (size_t row = 0; row < posterior_output.rows; ++row)
		{
			double min, max;
			cv::minMaxLoc(output.row(row), &min, &max);
			output.at<uchar>(row, 0) = m_classes_[static_cast<uchar>(max)];
		}
	}

	void NaiveBayesClassifier::Posterior(const cv::Mat& input, cv::Mat& output) const
	{
		// Ensures this function isn't called on an untrained classifier.
		CheckIfTrained_();

		size_t features = input.cols;
		if (features < 1 || features != m_feature_classifiers_.size())
		{
			throw std::runtime_error("Amount of features don't align with trained features - trained: " + std::to_string(m_feature_classifiers_.size()) + " | input: " + std::to_string(features));
		}

		// Aggregate the probabilities for all features
		output = cv::Mat::zeros(input.rows, m_classes_.size(), CV_32FC1);
		output += 1.0;

		cv::Mat feature_output;
		for (size_t feature = 0; feature < features; ++feature)
		{
			for (size_t sample = 0; sample < input.rows; ++sample)
			{
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
		cv::reduce(output, norms, 1, cv::ReduceTypes::REDUCE_SUM, CV_64F);
		for (size_t row = 0; row < output.rows; ++row)
		{
			long double norm = norms.at<long double>(row, 0);

			if (norm == 0)
			{
				output.row(row).setTo(cv::Scalar(static_cast<float>(1) / static_cast<float>(m_classes_.size())));
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

	void NaiveBayesClassifier::Train(const cv::ml::TrainData& train_data)
	{
		Train(train_data, std::vector<std::string>());
	}

	void NaiveBayesClassifier::Train(const cv::ml::TrainData& train_data, const std::vector<std::string> feature_names)
	{
		if (train_data.getLayout() != cv::ml::ROW_SAMPLE)
		{
			throw std::runtime_error("The classifier requires row based samples.");
		}

		if (!feature_names.empty())
		{
			m_trained_feature_names_ = feature_names;
		}
		else
		{
			cv::Mat sample = train_data.getSamples().row(0);

			for (size_t col = 0; col < sample.cols; ++col)
			{
				m_trained_feature_names_.push_back(std::to_string(col));
			}
		}

		TrainClassifier_(train_data);
	}

	void NaiveBayesClassifier::CheckIfTrained_(void) const
	{
		if (!IsTrained())
		{
			throw std::runtime_error("Classifier must be trained before it can classify samples.");
		}
	}

	void NaiveBayesClassifier::TrainClassifier_(const cv::ml::TrainData& train_data)
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

			m_classes_.clear();
			for (const int32_t class_label : classes)
			{
				m_classes_.push_back(class_label);
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
			feature_classifier.Train(feature_samples, feature_responses, this->GetClasses().size(), n_bins, blur_sigma);
			m_feature_classifiers_.push_back(feature_classifier);
		}

		// Set the classifier as trained.
		m_is_trained_ = true;
	}
}