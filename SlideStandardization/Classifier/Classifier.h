/***************************************************************-*-c++-*-

@COPYRIGHT@

$Id: Classifier.h,v 1.12 2006/09/01 14:59:32 mirela Exp $

*************************************************************************/
//---------------------------------------------------------------------------
#ifndef __Classifier_H__
#define __Classifier_H__
//---------------------------------------------------------------------------

#include <vector>

#include <opencv2/ml.hpp>

/*! \class Classifier

\brief The base class for classifiers

\details
This class defines the interface for classifiers.

\author
Bran van Ginneken, Joes Staal

\see
DataSet
*/

namespace ML
{
	class Classifier
	{
		public:
			/// <summary>
			/// Default constructor
			/// </summary>
			Classifier(void);
			/// <summary>
			/// Copy constructor
			/// </summary>
			/// <param name="other">The classifier to copy contents from.</param>
			Classifier(const Classifier& other);
			/// <summary>
			/// Virtual destructor
			/// </summary>
			virtual ~Classifier(void);



		void SetWeights(const std::vector<double>& weights);
		const std::vector<double>& GetWeights(void) const;
		std::vector<double> CopyWeights(void);

		////////////////////
		// Returns the number of classes in the training set, returns -1 if
		// not trained or not implemented
		size_t GetNrOfClasses(void) const;

		///////////////////
		// Returns if the classifier is ready for use, that is, classify, regress,
		// posterior and pdf can be used (when implemented)
		bool IsTrained(void) const;

		////////////////////
		// Returns the feature names of the features used in training
		std::vector<std::string> GetFeatureNames(void) const;
		size_t GetNumberOfFeatures(void) const;

		//////////////////
		// Trains the classifier with the given data set. First the input mappings are
		// trained and applied to the dataset, subsequently trainClassifier is called on the
		// transformed dataset.
		void Train(const cv::ml::TrainData& train_data);
		void Train(const cv::ml::TrainData& train_data, const std::vector<std::string> feature_names);

		//////////////////
		// Hard classification of the sample s into one of the possible output classes.
		void Classify(const cv::Mat& input, cv::Mat& output) const;

		//////////////////
		// Regression of sample s onto one of the outputs.
		//void Regress(const cv::Mat& input, cv::Mat& output) const;




		/////////////////
		// Soft classification of the sample s.  Posterior probability for each
		// possible output class is determined.
		// Posterior probabilities should add up to one.
		// N.B. number of outputs equals number of classes.
		void Posterior(const cv::Mat& input, cv::Mat& output) const;

		//////////////////
		// Probability density function at this sample. This value is linearly
		// related to the chance that this sample is drawn from the distribution modeled
		// by this classifier object.
		// This function assumes the input samples model a single pdf, so the output
		// of each sample is not used.
		//void Pdf(const cv::Mat& input, cv::Mat& output) const;

	protected:
		bool						m_is_trained$;
		std::vector<uchar>			m_classes$;
		std::vector<std::string>	m_trained_feature_names$;
		std::vector<double>			m_weights$;

		/// <summary>
		/// Trains the classifier with the given data set.
		/// </summary>
		/// <param name="input">The input matrix holding the samples to classify.</param>
		/// <param name="output">The output matrix to write the results into.</param>
		virtual void TrainClassifier$(const cv::ml::TrainData& train_data) = 0;
		/// <summary>
		/// Classifies the samples.
		/// </summary>
		/// <param name="input">The input matrix holding the samples to classify.</param>
		/// <param name="output">The output matrix to write the results into.</param>
		virtual void ClassifySamples$(const cv::Mat& input, cv::Mat& output) const = 0;
		/// <summary>
		/// Regresses the samples.
		/// </summary>
		/// <param name="input">The input matrix holding the samples to classify.</param>
		/// <param name="output">The output matrix to write the results into.</param>
		//virtual void RegressSamples$(const cv::Mat& input, cv::Mat& output) const = 0;
		/// <summary>
		/// Performs a soft classification of the samples, providing a posterior probability for each
		/// possible class output. These are added together and should reach one.
		/// The number of outputs per sample equal the amount of classes.
		/// </summary>
		/// <param name="input">The input matrix holding the samples to classify.</param>
		/// <param name="output">The output matrix to write the results into.</param>
		virtual void PosteriorSamples$(const cv::Mat& input, cv::Mat& output) const = 0;
		/// <summary>
		/// Calculates the probability density for the provided samples.
		/// </summary>
		/// <param name="input">The input matrix holding the samples to classify.</param>
		/// <param name="output">The output matrix to write the results into.</param>
		//virtual void PdfSamples$(const cv::Mat& input, cv::Mat& output) const = 0;

	private:
		/// <summary>
		/// Checks if the classifier can be used, throws a runtime exception if it can't.
		/// </summary>
		void CheckIfTrained_(void) const;
	};
}
#endif // __Classifier_H__