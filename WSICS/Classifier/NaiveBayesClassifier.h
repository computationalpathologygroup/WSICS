#ifndef __NaiveBayesClassifier_H__
#define __NaiveBayesClassifier_H__
//---------------------------------------------------------------------------
#include "Classifier.h"
#include "NaiveBayesFeatureClassifier.h"


/*! \class NaiveBayesClassifier

\brief Helper class for the Naive Bayesian classifier

\details
This simple classifier, aka the Idiot Bayes Classifier, assumes that all
features are independent. It computes a probability table for each feature
and when classifying a sample it simply multiplies the probabilities per
feature (and renormalizes the resulting probabilities per class).

See Duda and Hart 2nd edition page 61.

There are two free parameters. One free parameter is the number of bins in
each probability table. A histogram is computed for each feature with the
given number of bins between min and max.
This histogram may be blurred. Another free parameter controls this.

IMPLEMENTED CLASSIFICATIONS

<ul>
<li>classify</li>
<li>posterior</li>
</ul>

\author

Bram van Ginneken
*/
namespace WSICS::ML
{
	class NaiveBayesClassifier : public Classifier
	{
		public:
			enum ProbabilityIntegration { ADDITION, MULTIPLICATION };
			
			size_t					n_bins;
			float					blur_sigma;
			ProbabilityIntegration	probability_integration_type;

			/////////////////
			// Default constructor
			NaiveBayesClassifier(const size_t bins = 100, const float blur = 3, const ProbabilityIntegration integration_type = MULTIPLICATION);

			// Copy constructor
			NaiveBayesClassifier(const NaiveBayesClassifier& other);

		protected:
			void TrainClassifier$(const cv::ml::TrainData& train_data);
			void ClassifySamples$(const cv::Mat& input, cv::Mat& output) const;
			void PosteriorSamples$(const cv::Mat& input, cv::Mat& output) const;

		private:
			std::vector<NaiveBayesFeatureClassifier>	m_feature_classifiers_;
	};
}
#endif // __NaiveBayesClassifier_H__