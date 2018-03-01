#ifndef __STANDARDIZATION_H__
#define __STANDARDIZATION_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

#include <boost/system/config.hpp>
#include <boost/program_options.hpp> 
#include <boost/filesystem.hpp>
#include <iostream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "HSD/HSD_Model.h"
#include "TransformCxCyDensity.h"
#include "PixelClassificationHE.h"

// TODO: Refactor into smaller, more well defined segments.


struct TransformationParameters
{
	MatrixRotationParameters hema_rotation_params;
	MatrixRotationParameters eosin_rotation_params;
	MatrixRotationParameters background_rotation_params;
	cv::Mat hema_scale_params;
	cv::Mat eosin_scale_params;
	ClassDensityRanges class_density_ranges;
};

class Standardization
{
	public:
		Standardization(std::string log_directory, const boost::filesystem::path& template_file, const uint32_t min_training_size, const uint32_t max_training_size);

		void Normalize(const boost::filesystem::path& input_file, const boost::filesystem::path& output_file, const boost::filesystem::path& template_output, const boost::filesystem::path& debug_directory, const bool consider_ink);
		void SetLogDirectory(std::string& log_directory);

	private:
		size_t							m_log_file_id_;
		boost::filesystem::path			m_debug_directory_;
		const boost::filesystem::path&	m_template_file_;

		uint32_t						m_min_training_size_;
		uint32_t						m_max_training_size_;
		bool							m_consider_ink_;
		bool							m_is_multiresolution_image_;

		cv::Mat									CalculateLutRawMat_(void);
		//cv::Mat									CreateNormalizedImage_(const HSD::HSD_Model& lut_hsd, const TrainingSampleInformation& sample_info, const boost::filesystem::path& output_file, const boost::filesystem::path& template_output);
		//TrainingSampleInformation				DownsampleforNbClassifier_(const TrainingSampleInformation& training_samples, const uint32_t downsample);		
		std::pair<bool, std::vector<double>>	GetResolutionTypeAndSpacing(MultiResolutionImage& tiled_image);
		std::vector<cv::Point>					GetTileCoordinates_(MultiResolutionImage& tiled_image, const std::vector<double>& spacing, const uint32_t tile_size, const uint32_t min_level);
		//void									HandleParameterization_(const TransformationParameters& calc_params, TransformationParameters& lut_params, const boost::filesystem::path& template_output);
		//void									PrintParameters_(std::ofstream& output_stream, const TransformationParameters& transform_param, const bool write_csv);
		//TransformationParameters				ReadParameters_(std::istream &input);

		TrainingSampleInformation CollectTrainingSamples_(
			const boost::filesystem::path& input_file,
			uint32_t tile_size,
			MultiResolutionImage& tiled_image,
			cv::Mat static_image,
			const std::vector<cv::Point>& tile_coordinates,
			const std::vector<double>& spacing,
			const uint32_t min_level);

		/*std::vector<cv::Mat> Standardization::InitializeTransformation_(
			const cv::Mat lut_cx_cy,
			const cv::Mat& cx_cy_train_data,
			const TransformationParameters& params,
			const TransformationParameters& transform_params,
			const ClassPixelIndices& class_pixel_indices);*/

		void WriteNormalizedWSI_(const boost::filesystem::path& input_file, const boost::filesystem::path& output_file, const cv::Mat& normalized_lut, const uint32_t tile_size);
		void WriteSampleNormalizedImagesForTesting_(const std::string output_filename, const cv::Mat& normalized_lut, const cv::Mat& tile_image, const uint32_t tile_size);
		void WriteSampleNormalizedImagesForTesting_(const boost::filesystem::path& output_directory, const cv::Mat& lut_image, MultiResolutionImage& tiled_image, const std::vector<cv::Point>& tile_coordinates, const uint32_t tile_size);
};
#endif // __STANDARDIZATION_H__