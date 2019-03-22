#ifndef __WSICS_CLASSIFICATION_NAIVEBAYSECLASSIFIER__
#define __WSICS_CLASSIFICATION_NAIVEBAYSECLASSIFIER__

#include <cstdint>

#include <opencv2/ml.hpp>

#include "NaiveBayesFeatureClassifier.h"

namespace WSICS::ML
{
	class NaiveBayesClassifier
	{
		public:
			enum ProbabilityIntegration { ADDITION, MULTIPLICATION };
			
			uint64_t				n_bins;
			float					blur_sigma;
			ProbabilityIntegration	probability_integration_type;

			/// <summary>
			/// Default constructor
			/// </summary>
			NaiveBayesClassifier(const uint64_t bins = 100, const float blur = 3, const ProbabilityIntegration integration_type = MULTIPLICATION);

			const std::vector<uchar> GetClasses(void) const;
			const std::vector<std::string>& GetFeatureNames(void) const;
			const std::vector<double>& GetWeights(void) const;
			void SetWeights(const std::vector<double>& weights);
			
			/// <summary>
			/// Returns whether or not the classifier has been trained.
			/// </summary>
			/// <returns>Whether or not the classifier has been trained.</returns>
			bool IsTrained(void) const;

			/// <summary>
			/// Trains the classifier with the given data set.
			/// </summary>
			/// <param name="input">The input matrix holding the samples to classify.</param>
			/// <param name="output">The output matrix to write the results into.</param>
			void Train(const cv::ml::TrainData& train_data);
			/// <summary>
			/// Trains the classifier with the given data set.
			/// </summary>
			/// <param name="input">The input matrix holding the samples to classify.</param>
			/// <param name="output">The output matrix to write the results into.</param>
			void Train(const cv::ml::TrainData& train_data, const std::vector<std::string> feature_names);

			/// <summary>
			/// Performs a hard classification the samples.
			/// </summary>
			/// <param name="input">The input matrix holding the samples to classify.</param>
			/// <param name="output">The output matrix to write the results into.</param>
			void Classify(const cv::Mat& input, cv::Mat& output) const;

			/// <summary>
			/// Performs a soft classification of the samples, providing a posterior probability for each
			/// possible class output. These are added together and should reach one.
			/// The number of outputs per sample equal the amount of classes.
			/// </summary>
			/// <param name="input">The input matrix holding the samples to classify.</param>
			/// <param name="output">The output matrix to write the results into.</param>
			void Posterior(const cv::Mat& input, cv::Mat& output) const;

		private:
			bool										m_is_trained_;
			std::vector<uchar>							m_classes_;
			std::vector<std::string>					m_trained_feature_names_;
			std::vector<double>							m_weights_;
			std::vector<NaiveBayesFeatureClassifier>	m_feature_classifiers_;

			/// <summary>
			/// Checks if the classifier can be used, throws a runtime exception if it can't.
			/// </summary>
			void CheckIfTrained_(void) const;

			void TrainClassifier_(const cv::ml::TrainData& train_data);
	};
}
#endif // __WSICS_CLASSIFICATION_NAIVEBAYSECLASSIFIER__