#ifndef __PixelClassificationHE_H__
#define __PixelClassificationHE_H__

#include <opencv2/core/core.hpp>

#include "HE_Staining/HE_Classifier.h"
#include "HSD/HSD_Model.h"
#include "MaskGeneration.h"
#include "multiresolutionimageinterface/MultiResolutionImage.h"

typedef HE_Staining::ClassificationResults ClassificationResults;
typedef HE_Staining::EosinMaskInformation EosinMaskInformation;
typedef HE_Staining::HematoxylinMaskInformation HematoxylinMaskInformation;

struct SampleInformation
{
	cv::Mat training_data_c_x;
	cv::Mat training_data_c_y;
	cv::Mat training_data_density;
	cv::Mat class_data;
};

class PixelClassificationHE
{
	public:
		PixelClassificationHE(bool consider_ink, size_t log_file_id, std::string& debug_dir);

		SampleInformation GenerateCxCyDSamples(
			MultiResolutionImage& tile_reader,
			cv::Mat& static_image,
			std::vector<cv::Point>& tile_coordinates,
			uint32_t tile_size,
			size_t training_size,
			size_t min_training_size,
			uint32_t min_level,
			float hema_percentile,
			float eosin_percentile,
			bool is_tiff,
			std::vector<double>& spacing);

	private:
		bool		m_consider_ink_;
		size_t		m_log_file_id_;
		std::string m_debug_dir_;

		std::pair<HematoxylinMaskInformation, EosinMaskInformation> Create_HE_Masks_(
			const HSD::HSD_Model& hsd_image,
			const cv::Mat& background_mask,
			const size_t min_training_size,
			const float hema_percentile,
			const float eosin_percentile,
			const std::vector<double>& spacing,
			const bool is_multiresolution);


		/// <param name="total_hema_count">The current total count of the Hematoxylin pixels. Warning: This methods updates the value with newly discovered pixels.</param>
		/// <param name="total_eosin_count">The current total count of the Eosin pixels. Warning: This methods updates the value with newly discovered pixels.</param>
		/// <param name="total_background_count">The current total count of the background pixels. Warning: This methods updates the value with newly discovered pixels.</param>
		SampleInformation InsertTrainingData_(
			const HSD::HSD_Model& hsd_image,
			const ClassificationResults& classification_results,
			const HematoxylinMaskInformation& hema_mask_info,
			const EosinMaskInformation& eosin_mask_info,
			SampleInformation& sample_information,
			size_t& total_hema_count,
			size_t& total_eosin_count,
			size_t& total_background_count,
			const size_t training_size);

		SampleInformation PatchTestData_(const size_t non_zero_count, const SampleInformation& current_sample_information);
};
#endif // __PixelClassificationHE_H__