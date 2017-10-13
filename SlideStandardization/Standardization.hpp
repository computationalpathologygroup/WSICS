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
		bool is_multiresolution_image;

		Standardization(std::string log_directory, bool is_multiresolution_image = false);

		void CreateNormalizationLUT(
			std::string& input_file,
			std::string& parameters_location,
			std::string& output_directory,
			std::string& debug_directory,
			size_t training_size,
			size_t min_training_size,
			uint32_t tile_size,
			bool is_tiff,
			bool only_generate_parameters,
			bool consider_ink);

		void SetLogDirectory(std::string& log_directory);
		void WriteNormalizedWSI(std::string& slide_directory, cv::Mat& normalized_lut, uint32_t tile_size, bool is_tiff, std::string& output_filepath);

	private:
		size_t m_log_file_id_;

		cv::Mat CalculateLutRawMat_(void);
		cv::Mat CreateNormalizedImage_(HSD::HSD_Model& hsd_lut, SampleInformation& sample_info, size_t training_size, std::string& parameters_filepath, bool only_generate_parameters);
		SampleInformation DownsampleforNbClassifier_(SampleInformation& sample_information, uint32_t downsample, size_t training_size);
		std::vector<cv::Point> GetTileCoordinates_(std::string& slide_directory, std::vector<double>& spacing, uint32_t tile_size, bool is_tiff, bool is_multi_resolution);

		void HandleParameterization(TransformationParameters& calc_params, TransformationParameters& lut_params, std::string& parameters_filepath);

		std::vector<cv::Mat> Standardization::InitializeTransformation_(
			HSD::HSD_Model& hsd_lut,
			cv::Mat& cx_cy_train_data,
			TransformationParameters& params,
			TransformationParameters& transform_params,
			ClassPixelIndices& class_pixel_indices);

		void PrintParameters_(std::ofstream& output_stream, TransformationParameters& transform_param, bool write_csv);
		TransformationParameters ReadParameters_(std::istream &input);

		void WriteSampleNormalizedImagesForTesting_(cv::Mat& normalized_lut, cv::Mat& tile_image, uint32_t tile_size, bool is_tiff, std::string& output_directory);
		void WriteSampleNormalizedImagesForTesting_(cv::Mat& normalized_image, MultiResolutionImage& tiled_image, uint32_t tile_size, std::vector<cv::Point>& tile_coordinates, bool is_tiff, std::string& debug_dir, std::string& filename);
};
#endif // __STANDARDIZATION_H__