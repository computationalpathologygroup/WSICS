#include "NaiveBayesFeatureClassifier.h"

#include <stdexcept>
#include <float.h>
#include <math.h>

namespace ML
{
	void NaiveBayesFeatureClassifier::Run(const float input, cv::Mat& output) const
	{
		output = cv::Mat::zeros(1, m_lut_.cols, CV_32FC1);
		// Get the position of the input in the histogram.
		float fbin = m_scale_ * (input - m_min_);
		if (fbin<0)
		{
			output = m_lut_.row(0);
		}
		else if (fbin >= (float)m_lut_.rows - 1)
		{
			output = m_lut_.row(m_lut_.rows - 1);
		}
		// Interpolate.
		else
		{
			int ibin = fbin;
			float a = fbin - ibin;
			output = (1 - a) * m_lut_.row(ibin);
			output += a * m_lut_.row(ibin + 1);
		}
	}

	/*void NaiveBayesFeatureClassifier::Process(const cv::Mat& input, cv::Mat output) const
	{
		output = cv::Mat::zeros(input.rows, m_lut_.cols, CV_32FC1);


		cv::Mat fbin
		if (input.data != output.data)
		{
			input.copyTo(output);
		}
		
		output =

	}*/

	// Computes a (Gaussian-blurred) histogram of p(x|class) for each class, p(x)
	// (histogram of all classes) and p(class) (the priors). With these quantities
	// we can apply Bayes' rule: p(class|x) = p(x|class) p(class) / p(x).
	void NaiveBayesFeatureClassifier::Train(const std::vector<float>& samples, std::vector<int32_t>& responses, size_t nrclasses,
		size_t bins, float sigma)
	{
		if (!(samples.size()>0 && responses.size() == samples.size() && bins>0 && sigma>0))
		{
			throw std::runtime_error("Not all parameters are correct.");
		}

		float sigmainv = 1.0 / sigma;

		// Compute the minimum and maximum values of the feature.
		m_min_ = *std::min_element(samples.begin(), samples.end());
		float max = *std::max_element(samples.begin(), samples.end());

		// Determine the bin size.
		m_n_bins_ = bins;
		float binsize = (max - m_min_) / bins;
		assert(binsize>0);
		m_scale_ = 1.0 / binsize;

		m_lut_ = cv::Mat::zeros(bins, nrclasses, CV_32FC1);

		// Initialize the p(x|class) LUT.
		std::vector<float> priors(nrclasses), px(m_lut_.rows);

		// Compute the blurred histogram (with "Gaussian bins") of p(x|class)
		// and the priors.
		for (size_t i = 0; i<samples.size(); ++i)
		{
			int current_class = responses[i];
			priors[current_class]++;
			float f = (samples[i] - m_min_) * m_scale_; // histogram index
			int minindex = std::max((int)0, int(f - 2.5*sigma));
			int maxindex = std::min((int)bins, int(f + 2.5*sigma + 1));
			for (int j = minindex; j<maxindex; ++j)
			{
				float d = (j - f)*sigmainv; // blurred histogram index
				float fac = exp(-d*d); // Gaussian bin count, max=1 when j==f

				m_lut_.at<float>(j, current_class) += fac;
				px[j] += fac;
			}
		}

		// Normalize the priors.
		for (size_t i = 0; i < nrclasses; ++i)
		{
			priors[i] /= samples.size();
		}

		// Normalize px.
		long double norm = 0.0;
		for (float val : px)
		{
			norm += val;
		}
		if (norm > 0)
		{
			for (float& val : px)
			{
				val /= norm;
			}
		}

		// Normalize the p(x|class) LUT.
		for (size_t class_index = 0; class_index < nrclasses; ++class_index)
		{
			long double norm = 0.0;
			for (size_t sample = 0; sample < m_lut_.rows; ++sample)
			{
				norm += m_lut_.at<float>(sample, class_index);
			}

			if (norm > 0)
			{
				for (size_t sample = 0; sample < m_lut_.rows; ++sample)
				{
					m_lut_.at<float>(sample, class_index) /= norm;
				}
			}
		}

		// Apply the Bayes' rule.
		for (size_t sample = 0; sample < m_lut_.rows; ++sample)
		{
			if (px[sample]>0.0f)
			{
				for (size_t class_index = 0; class_index < nrclasses; ++class_index)
				{
					// Apparently, breaking the statement instead of using *= solves the
					// problem with NaNs and Infs.
					float val = m_lut_.at<float>(sample, class_index) * priors[class_index] / px[sample];
					// NaN and Inf checking (just in case)
					if (val == val && val <= FLT_MAX && val >= -FLT_MAX)
					{
						m_lut_.at<float>(sample, class_index) = val;
					}
					else
					{
						m_lut_.at<float>(sample, class_index) = 0;
					}
				}
			}
		}
	}
}