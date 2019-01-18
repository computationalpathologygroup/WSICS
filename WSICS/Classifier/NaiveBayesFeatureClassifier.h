#ifndef __WSICS_CLASSIFICATION__NAIVEBAYESFEATURECLASSIFIER__
#define __WSICS_CLASSIFICATION__NAIVEBAYESFEATURECLASSIFIER__

#include <opencv2/core.hpp>
#include <vector>


namespace WSICS::ML
{
	class NaiveBayesFeatureClassifier
	{
		public:
			void Run(const float input, cv::Mat& output) const;
			//void Process(const cv::Mat& input, cv::Mat& output) const;
			void Train(const std::vector<float>& samples, std::vector<int32_t>& responses, const size_t nrclasses, const size_t bins, const float sigma);

		private:
			cv::Mat		m_lut_;
			float		m_min_, m_scale_;
			size_t		m_n_bins_;
	};
}
#endif // __WSICS_CLASSIFICATION__NAIVEBAYESFEATURECLASSIFIER__