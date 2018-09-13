/***************************************************************-*-c++-*-

@COPYRIGHT@

$Id: Classifier.cpp,v 1.11 2006/09/01 14:59:32 mirela Exp $

*************************************************************************/
//---------------------------------------------------------------------------

#include "Classifier.h"
#include <stdexcept>

namespace WSICS::ML
{
	Classifier::Classifier(void)
		: m_is_trained$(false)
	{
	}

	Classifier::Classifier(const Classifier& other)
		: m_is_trained$(other.m_is_trained$), m_classes$(other.m_classes$), m_trained_feature_names$(other.m_trained_feature_names$), m_weights$(other.m_weights$)
	{
	}

	Classifier::~Classifier(void)
	{
	}

	void Classifier::SetWeights(const std::vector<double>& weights)
	{
		m_weights$ = weights;
	}

	const std::vector<double>& Classifier::GetWeights(void) const
	{
		return m_weights$;
	}

	std::vector<double> Classifier::CopyWeights(void)
	{
		return m_weights$;
	}

	bool Classifier::IsTrained(void) const
	{
		return m_is_trained$;
	}

	std::vector<std::string> Classifier::GetFeatureNames(void) const
	{
		return m_trained_feature_names$;
	}

	size_t Classifier::GetNumberOfFeatures(void) const
	{
		return m_trained_feature_names$.size();
	}

	size_t Classifier::GetNrOfClasses(void) const
	{
		return m_classes$.size();
	}

	void Classifier::Train(const cv::ml::TrainData& train_data)
	{
		Train(train_data, std::vector<std::string>());
	}

	void Classifier::Train(const cv::ml::TrainData& train_data, const std::vector<std::string> feature_names)
	{
		if (train_data.getLayout() != cv::ml::ROW_SAMPLE)
		{
			throw std::runtime_error("The classifier requires row based samples.");
		}

		m_trained_feature_names$.clear();
		if (!feature_names.empty())
		{
			m_trained_feature_names$ = feature_names;
		}
		else
		{
			m_trained_feature_names$.clear();
			cv::Mat sample = train_data.getSamples().row(0);

			for (size_t col = 0; col < sample.cols; ++col)
			{
				m_trained_feature_names$.push_back(std::to_string(col));
			}
		}

		TrainClassifier$(train_data);
	}

	void Classifier::Classify(const cv::Mat& input, cv::Mat& output) const
	{
		CheckIfTrained_();
		ClassifySamples$(input, output);
	}

	void Classifier::Posterior(const cv::Mat& input, cv::Mat& output) const
	{
		CheckIfTrained_();
		PosteriorSamples$(input, output);
	}

	void Classifier::CheckIfTrained_(void) const
	{
		if (!IsTrained())
		{
			throw std::runtime_error("Classifier must be trained before it can classify samples.");
		}
	}
}