#ifndef __NORMALIZED_LUT_CREATION_H__
#define __NORMALIZED_LUT_CREATION_H__

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

#include "../HSD/HSD_Model.h"
#include "TransformCxCyDensity.h"
#include "PixelClassificationHE.h"

// TODO: Refactor into smaller, more well defined segments.

namespace WSICS::Normalization::NormalizedLutCreation
{
	struct TransformationParameters
	{
		TransformCxCyDensity::MatrixRotationParameters	hema_rotation_params;
		TransformCxCyDensity::MatrixRotationParameters	eosin_rotation_params;
		TransformCxCyDensity::MatrixRotationParameters	background_rotation_params;
		cv::Mat											hema_scale_params;
		cv::Mat											eosin_scale_params;
		TransformCxCyDensity::ClassDensityRanges		class_density_ranges;
	};

	cv::Mat	Create(
		const bool generate_lut,
		const boost::filesystem::path& template_file,
		const boost::filesystem::path& template_output,
		const HSD::HSD_Model& lut_hsd,
		const TrainingSampleInformation& training_sample,
		const uint32_t max_training_size,
		const size_t log_file_id);

	TrainingSampleInformation DownsampleforNbClassifier(const TrainingSampleInformation& training_samples, const uint32_t downsample, const uint32_t max_training_size);
	TransformationParameters HandleParameterization(const TransformationParameters& calc_params, const boost::filesystem::path& template_file, const boost::filesystem::path& template_output, const size_t log_file_id);
	std::vector<cv::Mat> InitializeTransformation(
		const cv::Mat& training_cx_cy,
		const cv::Mat& lut_cx_cy,
		const cv::Mat& cx_cy_hema_rotated,
		const cv::Mat& cx_cy_eosin_rotated,
		const TransformationParameters& params,
		const TransformationParameters& transform_params,
		const TransformCxCyDensity::ClassPixelIndices& class_pixel_indices);
	void PrintParameters(std::ofstream& output_stream, const TransformationParameters& transform_param, const bool write_csv);
	TransformationParameters ReadParameters(std::istream &input);


};
#endif // __NORMALIZED_LUT_CREATION_H__