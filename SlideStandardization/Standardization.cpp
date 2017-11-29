#include "Standardization.h"

#include <boost/filesystem.hpp>

#define _USE_MATH_DEFINES

#include <stdexcept>

#include <core/filetools.h>

#include "HSD/BackgroundMask.h"
#include "HSD/Transformations.h"

#include "CxCyWeights.h"
#include "LevelReading.h"
#include "IO/Logging/LogHandler.h"


#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImageWriter.h"
#include "multiresolutionimageinterface/OpenSlideImageFactory.h"

#include <math.h>

#include "MiscFunctionality.h"
#include <windows.h>
#include <strsafe.h>

Standardization::Standardization(std::string log_directory,	const boost::filesystem::path& template_file, const uint32_t min_training_size, const uint32_t max_training_size)
	: m_log_file_id_(0), m_template_file_(template_file), m_debug_directory_(), m_min_training_size_(min_training_size),
		m_max_training_size_(max_training_size), m_consider_ink_(false), m_is_multiresolution_image_(false), m_is_tiff_(false)
{
	this->SetLogDirectory(log_directory);
}


void Standardization::Normalize(
	const boost::filesystem::path& input_file,
	const boost::filesystem::path& output_file,
	const boost::filesystem::path& template_output,
	const boost::filesystem::path& debug_directory,
	const bool consider_ink,
	const bool is_tiff)
{
	//===========================================================================
	//	Sets several execution variables.
	//===========================================================================

	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	m_debug_directory_	= debug_directory;
	m_consider_ink_		= consider_ink;
	m_is_tiff_			= is_tiff;

	//===========================================================================
	//	Reading the image:: Identifying the tiles containing tissue using multiple magnifications
	//===========================================================================
	logging_instance->QueueFileLogging("=============================\n\nReading image...", m_log_file_id_, IO::Logging::NORMAL);

	MultiResolutionImageReader reader;
	MultiResolutionImage* tiled_image = reader.open(input_file.string());

	if (!tiled_image)
	{
		throw std::invalid_argument("Unable to open file: " + input_file.string());
	}

	// Acquires the type of image, the spacing and the minimum level to select tiles from.
	std::vector<double> spacing;
	uint32_t min_level = 0;

	// Scopes the pair so that it can be moved into more clearly defined variables.
	{
		std::pair<bool, std::vector<double>> resolution_and_spacing(GetResolutionTypeAndSpacing(*tiled_image));

		m_is_multiresolution_image_ = resolution_and_spacing.first;
		spacing.swap(resolution_and_spacing.second);
	}

	if (spacing[0] < 0.2)
	{
		spacing[0]	*= 2;
		min_level	= 1;
	}

	logging_instance->QueueFileLogging("Pixel spacing = " + std::to_string(spacing[0]), m_log_file_id_, IO::Logging::NORMAL);

	uint32_t tile_size = 512;
	cv::Mat static_image;
	std::vector<cv::Point> tile_coordinates;
	if (m_is_multiresolution_image_)
	{
		tile_coordinates.swap(GetTileCoordinates_(*tiled_image, spacing, tile_size, min_level));
	}
	else
	{
		logging_instance->QueueCommandLineLogging("Deconvolving patch image.", IO::Logging::NORMAL);
		static_image = cv::imread(input_file.string(), CV_LOAD_IMAGE_COLOR);
		tile_coordinates.push_back({ 0, 0 });
	}

	TrainingSampleInformation training_samples(CollectTrainingSamples_(input_file, tile_size, *tiled_image, static_image, tile_coordinates, spacing, min_level));

	logging_instance->QueueCommandLineLogging("sampling done!", IO::Logging::NORMAL);
	logging_instance->QueueFileLogging("=============================\nSampling done!", m_log_file_id_, IO::Logging::NORMAL);

	//===========================================================================
	//	Generating LUT Raw Matrix
	//===========================================================================
	logging_instance->QueueFileLogging("Defining LUT\nLUT HSD", m_log_file_id_, IO::Logging::NORMAL);
	HSD::HSD_Model hsd_lut(CalculateLutRawMat_(), HSD::CHANNEL_SHIFT);

	logging_instance->QueueFileLogging("LUT BG calculation", m_log_file_id_, IO::Logging::NORMAL);
	cv::Mat background_mask(HSD::BackgroundMask::CreateBackgroundMask(hsd_lut, 0.24, 0.22));

	//===========================================================================
	//	Normalizes the LUT.
	//===========================================================================
	cv::Mat normalized_lut(CreateNormalizedImage_(hsd_lut, training_samples, output_file, template_output));

	//===========================================================================
	//	Writing LUT image to disk
	//===========================================================================
	if (!output_file.empty())
	{
		std::string current_filepath = output_file.parent_path().string() + "/" + input_file.stem().string();
		std::string lut_output = current_filepath.substr(0, current_filepath.rfind("_normalized")) + "_lut.tif";

		logging_instance->QueueFileLogging("Writing LUT to: " + lut_output + " (this might take some time).", m_log_file_id_, IO::Logging::NORMAL);
		logging_instance->QueueCommandLineLogging("Writing LUT to: " + lut_output + " (this might take some time).", IO::Logging::NORMAL);

		cv::imwrite(lut_output, normalized_lut);

		//===========================================================================
		//	Write sample images to Harddisk For testing
		//===========================================================================
		// Don't remove! usable for looking at samples of standardization
		if (!m_debug_directory_.empty() && logging_instance->GetOutputLevel() == IO::Logging::DEBUG)
		{
			logging_instance->QueueFileLogging("Writing sample standardized images to: " + m_debug_directory_.string(), m_log_file_id_, IO::Logging::NORMAL);

			if (m_is_multiresolution_image_)
			{	
				WriteSampleNormalizedImagesForTesting_(boost::filesystem::path(m_debug_directory_.string() + "/normalized_examples"), normalized_lut, *tiled_image, tile_coordinates, tile_size);
			}
			else
			{
				boost::filesystem::path output_filepath(m_debug_directory_.string() + "/" + input_file.stem().string() + ".tif");
				WriteSampleNormalizedImagesForTesting_(output_filepath.string(), normalized_lut, static_image, tile_size);
			}
		}

		logging_instance->QueueFileLogging("Writing the standardized WSI in progress...", m_log_file_id_, IO::Logging::NORMAL);
		logging_instance->QueueCommandLineLogging("Writing the standardized WSI in progress...", IO::Logging::NORMAL);
		WriteNormalizedWSI_(input_file, output_file, normalized_lut, tile_size);
		logging_instance->QueueFileLogging("Finished writing the image.", m_log_file_id_, IO::Logging::NORMAL);
		logging_instance->QueueCommandLineLogging("Finished writing the image.", IO::Logging::NORMAL);
	}


	//===========================================================================
	//	Cleans execution variables
	//===========================================================================
	m_debug_directory_	= "";
	m_consider_ink_		= false;
	m_is_tiff_			= false;
}

void Standardization::SetLogDirectory(std::string& filepath)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	// Closes the current log file, if present.
	if (m_log_file_id_ > 0)
	{
		logging_instance->CloseFile(m_log_file_id_);
	}

	m_log_file_id_ = logging_instance->OpenFile(filepath, false);
}

TrainingSampleInformation Standardization::DownsampleforNbClassifier_(const TrainingSampleInformation& training_samples, const uint32_t downsample)
{
	TrainingSampleInformation sample_info_downsampled
	{
		cv::Mat::zeros(m_max_training_size_ / downsample, 1, CV_32FC1),
		cv::Mat::zeros(m_max_training_size_ / downsample, 1, CV_32FC1),
		cv::Mat::zeros(m_max_training_size_ / downsample, 1, CV_32FC1),
		cv::Mat::zeros(m_max_training_size_ / downsample, 1, CV_32FC1)
	};

	for (size_t downsampled_pixel = 0; downsampled_pixel < training_samples.class_data.rows / downsample; ++downsampled_pixel)
	{
		sample_info_downsampled.training_data_c_x.at<float>(downsampled_pixel, 0)		= training_samples.training_data_c_x.at<float>(downsampled_pixel * downsample, 0);
		sample_info_downsampled.training_data_c_y.at<float>(downsampled_pixel, 0)		= training_samples.training_data_c_y.at<float>(downsampled_pixel * downsample, 0);
		sample_info_downsampled.training_data_density.at<float>(downsampled_pixel, 0)	= training_samples.training_data_density.at<float>(downsampled_pixel * downsample, 0);
		sample_info_downsampled.class_data.at<float>(downsampled_pixel, 0)				= training_samples.class_data.at<float>(downsampled_pixel * downsample, 0);
	}

	return sample_info_downsampled;
}

cv::Mat Standardization::CalculateLutRawMat_(void)
{
	cv::Mat raw_lut(cv::Mat::zeros(256*256*256, 1, CV_8UC3));//16387064
	int counterTest  = 0;
	for (int i1=0; i1<256;++i1)
	{
		for (int i2=0; i2<256;++i2)
		{
			for (int i3=0; i3<256;++i3)
			{
				raw_lut.at<cv::Vec3b>(counterTest,0) = cv::Vec3b( i3, i2, i1);
				++counterTest;
			}
		}
	}
	return raw_lut;
}

TrainingSampleInformation Standardization::CollectTrainingSamples_(
	const boost::filesystem::path& input_file,
	uint32_t tile_size,
	MultiResolutionImage& tiled_image,
	cv::Mat static_image,
	const std::vector<cv::Point>& tile_coordinates,
	const std::vector<double>& spacing,
	const uint32_t min_level)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	//===========================================================================
	//	Performing Pixel Classification
	//===========================================================================
	std::string log_text = "Number of available tiles for stain sampling: " + std::to_string(tile_coordinates.size());
	logging_instance->QueueCommandLineLogging(log_text, IO::Logging::NORMAL);
	logging_instance->QueueFileLogging(log_text, m_log_file_id_, IO::Logging::NORMAL);

	PixelClassificationHE pixel_classification_he(m_consider_ink_, m_log_file_id_, m_debug_directory_.string());
	float hema_percentile = 0.1; // The higher the value, the more conservative the classifier becomes in picking up blue, so standardization will also be pinkish - breast 0.1
	float eosin_percentile = 0.2;
	tile_size = 2048;

	uint32_t min_training_size = 0;
	if (m_is_multiresolution_image_)
	{
		min_training_size = m_min_training_size_;
	}

	TrainingSampleInformation training_samples(pixel_classification_he.GenerateCxCyDSamples(
		tiled_image,
		static_image,
		tile_coordinates,
		spacing,
		tile_size,
		min_training_size,
		m_max_training_size_,
		min_level,
		hema_percentile,
		eosin_percentile,
		m_is_multiresolution_image_,
		m_is_tiff_));

	return training_samples;
}

cv::Mat Standardization::CreateNormalizedImage_(const HSD::HSD_Model& hsd_lut, const TrainingSampleInformation& training_samples, const boost::filesystem::path& output_file, const boost::filesystem::path& template_output)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	//===========================================================================
	//	Transforming Cx and Cy distributions - Initialization 1
	//  Extracting all the parameters needed for applying the transformation
	//===========================================================================
	logging_instance->QueueFileLogging("Defining variables for transformation...", m_log_file_id_, IO::Logging::NORMAL);

	ClassAnnotatedCxCy train_data(TransformCxCyDensity::ClassCxCyGenerator(training_samples.class_data, training_samples.training_data_c_x, training_samples.training_data_c_y));
	
	// Rotates the cx_cy matrice per class and stores the parameters used.
	cv::Mat cx_cy_hema_rotated;
	cv::Mat cx_cy_eosin_rotated;
	cv::Mat cx_cy_background_Rotated;

	MatrixRotationParameters hema_rotation_info(TransformCxCyDensity::RotateCxCy(train_data.cx_cy_merged, cx_cy_hema_rotated, train_data.hema_cx_cy));
	MatrixRotationParameters eosin_rotation_info(TransformCxCyDensity::RotateCxCy(train_data.cx_cy_merged, cx_cy_eosin_rotated, train_data.eosin_cx_cy));
	MatrixRotationParameters background_rotation_info(TransformCxCyDensity::RotateCxCy(train_data.cx_cy_merged, cx_cy_background_Rotated, train_data.background_cx_cy));

	ClassPixelIndices class_pixel_indices(TransformCxCyDensity::GetClassIndices(training_samples.class_data));

	// Calculates the scale parameters per class.
	cv::Mat hema_scale_parameters(TransformCxCyDensity::CalculateScaleParameters(class_pixel_indices.hema_indices, cx_cy_hema_rotated));
	cv::Mat eosin_scale_parameters(TransformCxCyDensity::CalculateScaleParameters(class_pixel_indices.eosin_indices, cx_cy_eosin_rotated));
	cv::Mat background_scale_parameters(TransformCxCyDensity::CalculateScaleParameters(class_pixel_indices.background_indices, cx_cy_background_Rotated));

	ClassDensityRanges class_density_ranges(TransformCxCyDensity::GetDensityRanges(training_samples.class_data, training_samples.training_data_density, class_pixel_indices));

	logging_instance->QueueFileLogging("Finished computing tranformation parameters for the current image", m_log_file_id_, IO::Logging::NORMAL);

	// If no output directory is set, assume only template generation is required.
	if (output_file.empty())
	{
		return cv::Mat();
	}

	//===========================================================================
	//	Generating the weights for each class
	//===========================================================================
	// Downsample the number of samples for NB classifier
	uint32_t downsample = 20;
	if (m_max_training_size_ > 10000000 && m_max_training_size_ < 20000000)
	{
		downsample = 30;
	}
	else if (m_max_training_size_ >= 20000000 && m_max_training_size_ < 30000000)
	{
		downsample = 40;
	}
	else if (m_max_training_size_ >= 30000000)
	{
		downsample = 50;
	}

	logging_instance->QueueFileLogging("Down sampling the data for constructing NB classifier", m_log_file_id_, IO::Logging::NORMAL);

	TrainingSampleInformation sample_info_downsampled(DownsampleforNbClassifier_(training_samples, downsample));

	logging_instance->QueueFileLogging("Generating weights with NB classifier", m_log_file_id_, IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Generating the weights, Setting dataset of size " + std::to_string(sample_info_downsampled.class_data.rows * sample_info_downsampled.class_data.cols),
		IO::Logging::NORMAL);

	cv::Ptr<cv::ml::NormalBayesClassifier> classifier(CxCyWeights::CreateNaiveBayesClassifier(	sample_info_downsampled.training_data_c_x,	
																								sample_info_downsampled.training_data_c_y, 
																								sample_info_downsampled.training_data_density, 
																								sample_info_downsampled.class_data));
	logging_instance->QueueCommandLineLogging("Training Naive Bayes Classifier fininshed...", IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("Generating posteriors (This will take some time...)", IO::Logging::NORMAL);
	CxCyWeights::Weights weights(CxCyWeights::GenerateWeights(hsd_lut.c_x, hsd_lut.c_y, hsd_lut.density, classifier));
	logging_instance->QueueCommandLineLogging("All weights created...", IO::Logging::NORMAL);
	logging_instance->QueueFileLogging("Weights generated", m_log_file_id_, IO::Logging::NORMAL);

	//===========================================================================
	//	Defining Template Parameters
	//===========================================================================

	TransformationParameters calculated_transform_parameters{ hema_rotation_info, eosin_rotation_info, background_rotation_info, hema_scale_parameters, eosin_scale_parameters, class_density_ranges };
	TransformationParameters lut_transform_parameters(calculated_transform_parameters);

	HandleParameterization_(calculated_transform_parameters, lut_transform_parameters, template_output);

	//===========================================================================
	//	Transforming Cx and Cy distributions - Initialization
	//===========================================================================
	logging_instance->QueueCommandLineLogging("Transformation started...", IO::Logging::NORMAL);
	logging_instance->QueueFileLogging("Transformation started...", m_log_file_id_, IO::Logging::NORMAL);

	std::vector<cv::Mat> lut_transformation_results(InitializeTransformation_(hsd_lut, train_data.cx_cy_merged, calculated_transform_parameters, lut_transform_parameters, class_pixel_indices));

	//===========================================================================
	//	Generating the weights for each class
	//===========================================================================
	logging_instance->QueueFileLogging("Applying weights...", m_log_file_id_, IO::Logging::NORMAL);
	cv::Mat cx_cy_normalized(CxCyWeights::ApplyWeights(lut_transformation_results[0], lut_transformation_results[0], lut_transformation_results[1], weights));

	//===========================================================================
	//	Density scaling
	//===========================================================================
	//TransCxCyD.densityNormalization(trainDataAllD, HSD_LUT.Density, BG_Obj_LUT);
	logging_instance->QueueFileLogging("Density transformation...", m_log_file_id_, IO::Logging::NORMAL);
	cv::Mat density_scaling(TransformCxCyDensity::DensityNormalizationThreeScales(calculated_transform_parameters.class_density_ranges, lut_transform_parameters.class_density_ranges, hsd_lut.density, weights));

	//===========================================================================
	//	HSD reverse
	//===========================================================================
	logging_instance->QueueFileLogging("HSD reverse...", m_log_file_id_, IO::Logging::NORMAL);
	cv::Mat nornmalized_image_rgb;
	HSD::CxCyToRGB(cx_cy_normalized, nornmalized_image_rgb, density_scaling);

	return nornmalized_image_rgb;
}

std::pair<bool, std::vector<double>> Standardization::GetResolutionTypeAndSpacing(MultiResolutionImage& tiled_image)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
	std::pair<bool, std::vector<double>> resolution_and_spacing(true, tiled_image.getSpacing());

	if (!resolution_and_spacing.second.empty())
	{
		if (resolution_and_spacing.second[0] > 1)
		{
			resolution_and_spacing.first = false;
			resolution_and_spacing.second.clear();
			logging_instance->QueueFileLogging("Image is static!", m_log_file_id_, IO::Logging::NORMAL);
		}
		else
		{
			logging_instance->QueueFileLogging("Image is multi-resolution", m_log_file_id_, IO::Logging::NORMAL);
		}
	}

	if (resolution_and_spacing.second.empty())
	{
		logging_instance->QueueCommandLineLogging("The image does not have spacing information. Continuing with the default 0.24.", IO::Logging::NORMAL);
		resolution_and_spacing.second.push_back(0.243);
		logging_instance->QueueFileLogging("The image does not have spacing information. Continuing with the default 0.24. Pixel spacing set to default = " + std::to_string(resolution_and_spacing.second[0]), m_log_file_id_, IO::Logging::NORMAL);
	}

	return resolution_and_spacing;
}

std::vector<cv::Point> Standardization::GetTileCoordinates_(MultiResolutionImage& tiled_image, const std::vector<double>& spacing, const uint32_t tile_size, const uint32_t min_level)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	unsigned char number_of_levels = tiled_image.getNumberOfLevels();
	if (number_of_levels > 5)
	{
		number_of_levels = 5;
	}

	logging_instance->QueueFileLogging("Number of levels available = " + std::to_string(number_of_levels), m_log_file_id_, IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("detecting tissue regions...", IO::Logging::NORMAL);
	logging_instance->QueueFileLogging("Detecting tissue", m_log_file_id_, IO::Logging::NORMAL);

	// Attempts to acquire the tile coordinates for the lowest level / highest magnification.
	std::vector<size_t> dimensions		= tiled_image.getLevelDimensions(number_of_levels - 1);
	uint32_t skip_factor				= 1;
	float background_tissue_threshold	= 0.9;
	uint32_t level_scale_difference		= 1;

	std::vector<cv::Point> tile_coordinates;
	if (number_of_levels > 1)
	{
		logging_instance->QueueCommandLineLogging("Analyzing level: " + std::to_string(number_of_levels - 1), IO::Logging::NORMAL);

		std::vector<unsigned long long> next_level_dimensions = tiled_image.getLevelDimensions(number_of_levels - 2);
		level_scale_difference = std::pow(std::round(next_level_dimensions[0] / next_level_dimensions[0]), 2);

		// Loops through each level, acquiring coordinates for each and reusing them to calculate the set of coordinates for a higher magnification.
		tile_coordinates.swap(LevelReading::ReadLevelTiles(tiled_image, dimensions[0], dimensions[1], tile_size, number_of_levels - 1, skip_factor, background_tissue_threshold, m_is_tiff_));
		for (char level_number = number_of_levels - 2; level_number >= 0; --level_number)
		{
			if (level_number != 0)
			{
				skip_factor = 2;

				std::vector<size_t> dimensions_level_low	= tiled_image.getLevelDimensions(level_number);
				std::vector<size_t> dimensions_level_high	= tiled_image.getLevelDimensions(level_number - 1);
				level_scale_difference = std::pow(std::floor(dimensions_level_high[0] / dimensions_level_low[0]), 2);
			}

			std::string log_text = "Analyzing level: " + std::to_string(level_number) + " - Tiles containing tissue: " + std::to_string(tile_coordinates.size());
			logging_instance->QueueCommandLineLogging(log_text, IO::Logging::NORMAL);
			logging_instance->QueueFileLogging(log_text, m_log_file_id_, IO::Logging::NORMAL);

			background_tissue_threshold -= 0.1;
			tile_coordinates.swap(LevelReading::ReadLevelTiles(tiled_image, tile_coordinates, tile_size, level_number, skip_factor, level_scale_difference, background_tissue_threshold, m_is_tiff_));
		}
	}
	else
	{
		tile_coordinates.swap(LevelReading::ReadLevelTiles(tiled_image, dimensions[0], dimensions[1], tile_size, m_is_tiff_, number_of_levels - 1, 0.9, skip_factor));
	}

	return tile_coordinates;
}

void Standardization::HandleParameterization_(const TransformationParameters& calc_params, TransformationParameters& lut_params, const boost::filesystem::path& template_output)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	if (!m_template_file_.empty())
	{
		logging_instance->QueueFileLogging("Loading Template parameters...", m_log_file_id_, IO::Logging::NORMAL);
		std::ifstream csv_input_stream;
		csv_input_stream.open(m_template_file_.string());
		if (!csv_input_stream)
		{
			logging_instance->QueueCommandLineLogging("Could not read CSV file!", IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Could not read template CSV file!", m_log_file_id_, IO::Logging::NORMAL);

			throw std::runtime_error("Could not read CSV file.");
		}
		else
		{
			lut_params = ReadParameters_(csv_input_stream);
			csv_input_stream.close();
		}
	}
	else if (!template_output.empty())
	{
		logging_instance->QueueFileLogging("Saving Template parameters...", m_log_file_id_, IO::Logging::NORMAL);
		std::ofstream csv_output_stream;
		csv_output_stream.open(template_output.string());
		if (csv_output_stream)
		{
			PrintParameters_(csv_output_stream, calc_params, true);
			csv_output_stream.close();

			logging_instance->QueueCommandLineLogging("Done", IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Template Parameters written to: " + template_output.string(), m_log_file_id_, IO::Logging::NORMAL);
		}
		else
		{
			logging_instance->QueueCommandLineLogging("Could not write template CSV file!", IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Could not write template CSV file!", m_log_file_id_, IO::Logging::NORMAL);
		}
	}
}

std::vector<cv::Mat> Standardization::InitializeTransformation_(
	const HSD::HSD_Model& hsd_lut,
	const cv::Mat& cx_cy_train_data,
	const TransformationParameters& params,
	const TransformationParameters& transform_params,
	const ClassPixelIndices& class_pixel_indices)
{
	// Merges the c_x and c_y channels.
	cv::Mat cx_cy;
	cv::hconcat(std::vector<cv::Mat>({ hsd_lut.c_x, hsd_lut.c_y }), cx_cy);

	// Rotates the combined matrices for the hema and eosin classes.
	cv::Mat lut_hema_matrix;
	cv::Mat lut_eosin_matrix;
	TransformCxCyDensity::RotateCxCy(cx_cy, 
										lut_hema_matrix,
										transform_params.hema_rotation_params.x_median,
										transform_params.hema_rotation_params.y_median,
										transform_params.hema_rotation_params.angle);
	TransformCxCyDensity::RotateCxCy(cx_cy,
										lut_eosin_matrix,
										transform_params.eosin_rotation_params.x_median,
										transform_params.eosin_rotation_params.y_median,
										transform_params.eosin_rotation_params.angle);

	// Scales the rotated LUT_matrices
	TransformCxCyDensity::ScaleCxCyLUT(lut_hema_matrix,
										lut_hema_matrix,
										TransformCxCyDensity::AdjustParamaterMinMax(lut_hema_matrix, params.hema_scale_params),
										transform_params.hema_scale_params);
	TransformCxCyDensity::ScaleCxCy(lut_eosin_matrix,
										lut_eosin_matrix,
										TransformCxCyDensity::AdjustParamaterMinMax(lut_eosin_matrix, params.eosin_scale_params),
										transform_params.eosin_scale_params);

	// Scales the rotated calculated matrices.
/*	TransformCxCyDensity::ScaleCxCy(rotated_hema,
										rotated_hema,
										params.hema_scale_params,
										transform_params.hema_scale_params);
	TransformCxCyDensity::ScaleCxCy(rotated_eosin,
										rotated_eosin,
										params.eosin_scale_params,
										transform_params.eosin_scale_params);
										*/
	// Reverses the rotation.
	TransformCxCyDensity::RotateCxCyBack(lut_hema_matrix, lut_hema_matrix, transform_params.hema_rotation_params.angle - M_PI);
	TransformCxCyDensity::RotateCxCyBack(lut_eosin_matrix, lut_eosin_matrix, transform_params.eosin_rotation_params.angle - M_PI);

/*	TransformCxCyDensity::RotateCxCyBack(rotated_hema, rotated_hema, params.hema_rotation_params.angle);
	TransformCxCyDensity::RotateCxCyBack(rotated_eosin, rotated_eosin, params.eosin_rotation_params.angle);
	*/
	cv::Mat lut_background_matrix;
	TransformCxCyDensity::TranslateCxCyBack(lut_hema_matrix, lut_hema_matrix, class_pixel_indices.hema_indices, transform_params.hema_rotation_params.x_median, transform_params.hema_rotation_params.y_median);
	TransformCxCyDensity::TranslateCxCyBack(lut_eosin_matrix, lut_eosin_matrix, class_pixel_indices.eosin_indices, transform_params.eosin_rotation_params.x_median, transform_params.eosin_rotation_params.y_median);
	TransformCxCyDensity::TranslateCxCyBack(cx_cy_train_data, lut_background_matrix, class_pixel_indices.background_indices, transform_params.background_rotation_params.x_median, transform_params.background_rotation_params.y_median);

	return { lut_hema_matrix, lut_eosin_matrix, lut_background_matrix };
}

void Standardization::PrintParameters_(std::ofstream& output_stream, const TransformationParameters& transform_param, const bool write_csv)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	logging_instance->QueueCommandLineLogging("\nHematoxylin CxCy: "	+ std::to_string(transform_param.hema_rotation_params.x_median)			+ ", " + std::to_string(transform_param.hema_rotation_params.y_median)		+ "\n", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Eosin CxCy: "			+ std::to_string(transform_param.eosin_rotation_params.x_median)		+ ", " + std::to_string(transform_param.eosin_rotation_params.y_median)		+ "\n", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Background CxCy: "		+ std::to_string(transform_param.background_rotation_params.x_median)	+ ", " + std::to_string(transform_param.background_rotation_params.y_median)	+ "\n", IO::Logging::NORMAL);
	

	const cv::Mat& hema_scale_param(transform_param.hema_scale_params);
	const cv::Mat& eosin_scale_param(transform_param.eosin_scale_params);

	logging_instance->QueueCommandLineLogging("Hema Cx values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(hema_scale_param.at<float>(0, 0)) + " " + std::to_string(hema_scale_param.at<float>(1, 0)) + " " + std::to_string(hema_scale_param.at<float>(2, 0)) + " " +
		std::to_string(hema_scale_param.at<float>(3, 0)) + " " + std::to_string(hema_scale_param.at<float>(4, 0)) + " " + std::to_string(hema_scale_param.at<float>(5, 0)) + " " + std::to_string(hema_scale_param.at<float>(6, 0)), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\nHema Cy values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(hema_scale_param.at<float>(0, 1)) + " " + std::to_string(hema_scale_param.at<float>(1, 1)) + " " + std::to_string(hema_scale_param.at<float>(2, 1)) + " " +
		std::to_string(hema_scale_param.at<float>(3, 1)) + " " + std::to_string(hema_scale_param.at<float>(4, 1)) + " " + std::to_string(hema_scale_param.at<float>(5, 1)) + " " + std::to_string(hema_scale_param.at<float>(6, 1)), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\nEosin Cx values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(eosin_scale_param.at<float>(0, 0)) + " " + std::to_string(eosin_scale_param.at<float>(1, 0)) + " " + std::to_string(eosin_scale_param.at<float>(2, 0)) + " " +
		std::to_string(eosin_scale_param.at<float>(3, 0)) + " " + std::to_string(eosin_scale_param.at<float>(4, 0)) + " " + std::to_string(eosin_scale_param.at<float>(5, 0)) + " " + std::to_string(eosin_scale_param.at<float>(6, 0)), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\nEosin Cy values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(eosin_scale_param.at<float>(0, 1)) + " " + std::to_string(eosin_scale_param.at<float>(1, 1)) + " " + std::to_string(eosin_scale_param.at<float>(2, 1)) + " " +
		std::to_string(eosin_scale_param.at<float>(3, 1)) + " " + std::to_string(eosin_scale_param.at<float>(4, 1)) + " " + std::to_string(eosin_scale_param.at<float>(5, 1)) + " " + std::to_string(eosin_scale_param.at<float>(6, 1)), IO::Logging::NORMAL);


	logging_instance->QueueCommandLineLogging("Hema angle: " + std::to_string(transform_param.hema_rotation_params.angle), IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Eos angle: " + std::to_string(transform_param.eosin_rotation_params.angle), IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Background angle: " + std::to_string(transform_param.background_rotation_params.angle), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\n\nH&E Density Params: ", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(
		std::to_string(transform_param.class_density_ranges.hema_density_mean.val[0]) + ", " + std::to_string(transform_param.class_density_ranges.hema_density_standard_deviation.val[0]) + ", " +
		std::to_string(transform_param.class_density_ranges.eosin_density_mean.val[0]) + ", " + std::to_string(transform_param.class_density_ranges.eosin_density_standard_deviation.val[0]) + ", " +
		std::to_string(transform_param.class_density_ranges.background_density_mean.val[0]) + ", " + std::to_string(transform_param.class_density_ranges.background_density_standard_deviation.val[0]),
		IO::Logging::NORMAL);

	if (write_csv)
	{
		output_stream << transform_param.hema_rotation_params.x_median			<< "," << transform_param.hema_rotation_params.y_median			<< std::endl;
		output_stream << transform_param.eosin_rotation_params.x_median			<< "," << transform_param.eosin_rotation_params.y_median		<< std::endl;
		output_stream << transform_param.background_rotation_params.x_median	<< "," << transform_param.background_rotation_params.y_median	<< std::endl;

		output_stream << hema_scale_param.at<float>(0, 0) << ", " << hema_scale_param.at<float>(1, 0) << ", " << hema_scale_param.at<float>(2, 0) << ", " << hema_scale_param.at<float>(3, 0) << ", " <<
			hema_scale_param.at<float>(4, 0) << ", " << hema_scale_param.at<float>(5, 0) << ", " << hema_scale_param.at<float>(6, 0) << std::endl;
		output_stream << hema_scale_param.at<float>(0, 1) << ", " << hema_scale_param.at<float>(1, 1) << ", " << hema_scale_param.at<float>(2, 1) << ", " << hema_scale_param.at<float>(3, 1) << ", " <<
			hema_scale_param.at<float>(4, 1) << ", " << hema_scale_param.at<float>(5, 1) << ", " << hema_scale_param.at<float>(6, 1) << std::endl;

		output_stream << eosin_scale_param.at<float>(0, 0) << ", " << eosin_scale_param.at<float>(1, 0) << ", " << eosin_scale_param.at<float>(2, 0) << ", " << eosin_scale_param.at<float>(3, 0) << ", " <<
			eosin_scale_param.at<float>(4, 0) << ", " << eosin_scale_param.at<float>(5, 0) << ", " << eosin_scale_param.at<float>(6, 0) << std::endl;
		output_stream << eosin_scale_param.at<float>(0, 1) << ", " << eosin_scale_param.at<float>(1, 1) << ", " << eosin_scale_param.at<float>(2, 1) << ", " << eosin_scale_param.at<float>(3, 1) << ", " <<
			eosin_scale_param.at<float>(4, 1) << ", " << eosin_scale_param.at<float>(5, 1) << ", " << eosin_scale_param.at<float>(6, 1) << std::endl;

		output_stream << transform_param.hema_rotation_params.angle << std::endl;
		output_stream << transform_param.eosin_rotation_params.angle << std::endl;
		output_stream << transform_param.background_rotation_params.angle << std::endl;
	
		output_stream << transform_param.class_density_ranges.hema_density_mean.val[0] << ", " << transform_param.class_density_ranges.hema_density_standard_deviation.val[0] << ", " <<
			transform_param.class_density_ranges.eosin_density_mean.val[0] << ", " << transform_param.class_density_ranges.eosin_density_standard_deviation.val[0] << ", " <<
			transform_param.class_density_ranges.background_density_mean.val[0] << ", " << transform_param.class_density_ranges.background_density_standard_deviation.val[0] << std::endl;
	}
}

TransformationParameters Standardization::ReadParameters_(std::istream &input)
{
	std::vector<std::vector<float>> csv_parameters;

	std::string csv_line;
	// read every line from the stream
	int i = 0;
	while (std::getline(input, csv_line))
	{
		int j = 0;
		std::istringstream csvStream(csv_line);
		std::vector<float> csvColumn;
		std::string csvElement;
		std::vector<float> rowvals;
		std::vector<std::string> all_words;
		// read every element from the line that is seperated by commas
		// and put it into the vector or strings
		while (std::getline(csvStream, csvElement, ','))
		{
			rowvals.push_back(atof(csvElement.c_str()));
			j++;
		}
		csv_parameters.push_back(rowvals);
		i++;
	}

	TransformationParameters parameters;

	parameters.hema_rotation_params.x_median = csv_parameters[0][0];
	parameters.hema_rotation_params.y_median = csv_parameters[0][1];

	parameters.eosin_rotation_params.x_median = csv_parameters[1][0];
	parameters.eosin_rotation_params.y_median = csv_parameters[1][1];

	parameters.background_rotation_params.x_median = csv_parameters[2][0];
	parameters.background_rotation_params.y_median = csv_parameters[2][1];

	for (size_t i = 0; i < 7; ++i)
	{
		parameters.hema_scale_params.at<float>(i, 0) = csv_parameters[3][i];
		parameters.hema_scale_params.at<float>(i, 1) = csv_parameters[4][i];

		parameters.eosin_scale_params.at<float>(i, 0) = csv_parameters[5][i];
		parameters.eosin_scale_params.at<float>(i, 1) = csv_parameters[6][i];
	}

	// TODO: Find out why pi is getting subtracted from the imported angle.
	parameters.hema_rotation_params.angle = csv_parameters[7][0];
	parameters.eosin_rotation_params.angle = csv_parameters[8][0];
	parameters.background_rotation_params.angle = csv_parameters[9][0];

	parameters.class_density_ranges.hema_density_mean.val[0] = csv_parameters[10][0];
	parameters.class_density_ranges.hema_density_standard_deviation.val[0] = csv_parameters[10][1];
	parameters.class_density_ranges.eosin_density_mean.val[0] = csv_parameters[10][2];
	parameters.class_density_ranges.eosin_density_standard_deviation.val[0] = csv_parameters[10][3];
	parameters.class_density_ranges.background_density_mean.val[0] = csv_parameters[10][4];
	parameters.class_density_ranges.background_density_standard_deviation.val[0] = csv_parameters[10][5];

	return parameters;
}

void Standardization::WriteNormalizedWSI_(const boost::filesystem::path& input_file, const boost::filesystem::path& output_file, const cv::Mat& normalized_lut, const uint32_t tile_size)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	std::vector<cv::Mat> lut_bgr(3);
	cv::split(normalized_lut, lut_bgr);

	cv::Mat lut_blue	= lut_bgr[0];
	cv::Mat lut_green	= lut_bgr[1];
	cv::Mat lut_red		= lut_bgr[2];

	lut_blue.convertTo(lut_blue, CV_8UC1);
	lut_green.convertTo(lut_green, CV_8UC1);
	lut_red.convertTo(lut_red, CV_8UC1);

	MultiResolutionImageWriter image_writer;
	image_writer.openFile(output_file.string());
	image_writer.setTileSize(tile_size);
	image_writer.setCompression(pathology::LZW);
	image_writer.setDataType(pathology::UChar);
	image_writer.setColorType(pathology::RGB);
	image_writer.writeImageInformation(tile_size * 4, tile_size * 4);

	MultiResolutionImageReader reader;
	MultiResolutionImage* tiled_image	= reader.open(input_file.string());
	std::vector<size_t> dimensions		= tiled_image->getLevelDimensions(0);
	image_writer.writeImageInformation(dimensions[0], dimensions[1]);

	uint64_t y_amount_of_tiles = std::ceil((float)dimensions[0] / (float)tile_size);
	uint64_t x_amount_of_tiles = std::ceil((float)dimensions[1] / (float)tile_size);
	uint64_t total_amount_of_tiles = x_amount_of_tiles * y_amount_of_tiles;

	std::vector<uint64_t> x_values, y_values;

	for (uint64_t x_tile = 0; x_tile < x_amount_of_tiles; ++x_tile)
	{
		for (uint64_t y_tile = 0; y_tile < y_amount_of_tiles; ++y_tile)
		{
			y_values.push_back(tile_size*y_tile);
			x_values.push_back(tile_size*x_tile);
		}
	}

	size_t response_integer = total_amount_of_tiles / 20;
	for (uint64_t tile = 0; tile < total_amount_of_tiles; ++tile)
	{
		if (tile % response_integer == 0)
		{
			logging_instance->QueueCommandLineLogging("Completed: " + std::to_string((tile / response_integer) * 5) + "%", IO::Logging::NORMAL);
		}

		unsigned char* data = nullptr;
		tiled_image->getRawRegion(y_values[tile] * tiled_image->getLevelDownsample(0), x_values[tile] * tiled_image->getLevelDownsample(0), tile_size, tile_size, 0, data);

		std::vector<unsigned char> modified_data(tile_size * tile_size * 3);
		unsigned char* modified_data_ptr(&data[0]);

		size_t data_index = 0;
		size_t modified_data_index = 0;
		for (size_t pixel = 0; pixel < tile_size * tile_size; ++pixel)
		{
			size_t index = 256 * 256 * data[data_index] + 256 * data[data_index + 1] + data[data_index + 2];
			modified_data[modified_data_index++] = lut_red.at<unsigned char>(index, 0);
			modified_data[modified_data_index++] = lut_green.at<unsigned char>(index, 0);
			modified_data[modified_data_index++] = lut_blue.at<unsigned char>(index, 0);
			data_index += 3;
		}
		image_writer.writeBaseImagePart((void*)modified_data_ptr);

		delete[] data;
	}

	logging_instance->QueueCommandLineLogging("Finalizing images", IO::Logging::NORMAL);
	image_writer.finishImage();
}

void Standardization::WriteSampleNormalizedImagesForTesting_(const std::string output_filepath, const cv::Mat& normalized_lut, const cv::Mat& tile_image, const uint32_t tile_size)
{
	std::vector<cv::Mat> bgr_lut;
	cv::split(normalized_lut, bgr_lut);

	cv::Mat& blue_lut	= bgr_lut[0];
	cv::Mat& green_lut	= bgr_lut[1];
	cv::Mat& red_lut	= bgr_lut[2];

	blue_lut.convertTo(blue_lut, CV_8UC1);
	green_lut.convertTo(green_lut, CV_8UC1);
	red_lut.convertTo(red_lut, CV_8UC1);

	// get the BGR channels of tile image
	std::vector<cv::Mat> channels_in_tile(3);
	cv::split(tile_image, channels_in_tile);
	cv::Mat blue = channels_in_tile[0];
	cv::Mat green = channels_in_tile[1];
	cv::Mat red = channels_in_tile[2];

	for (size_t row = 1; row < tile_image.rows; ++row)
	{
		for (size_t col = 1; col < tile_image.cols; ++col)
		{
			size_t index = 256 * 256 * red.at<unsigned char>(row, col) + 256 * green.at<unsigned char>(row, col) + blue.at<unsigned char>(row, col);
			blue.at<unsigned char>(row, col) = blue_lut.at<unsigned char>(index, 0);
			green.at<unsigned char>(row, col) = green_lut.at<unsigned char>(index, 0);
			red.at<unsigned char>(row, col) = red_lut.at<unsigned char>(index, 0);
		}
	}

	cv::Mat lut_slide_image;
	cv::merge(channels_in_tile, lut_slide_image);

	cv::imwrite(output_filepath, lut_slide_image);

	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
	logging_instance->QueueCommandLineLogging("Normalized image is written...", IO::Logging::NORMAL);
}

void Standardization::WriteSampleNormalizedImagesForTesting_(
	const boost::filesystem::path& output_directory, 
	const cv::Mat& lut_image,
	MultiResolutionImage& tiled_image,
	const std::vector<cv::Point>& tile_coordinates,
	const uint32_t tile_size)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	logging_instance->QueueCommandLineLogging("Writing sample standardized images in: " + output_directory.string(), IO::Logging::NORMAL);

	std::vector<cv::Mat> lut_bgr(3);
	cv::split(lut_image, lut_bgr);

	cv::Mat& lut_blue	= lut_bgr[0];
	cv::Mat& lut_green	= lut_bgr[1];
	cv::Mat& lut_red	= lut_bgr[2];

	lut_red.convertTo(lut_red, CV_8UC1);
	lut_green.convertTo(lut_green, CV_8UC1);
	lut_blue.convertTo(lut_blue, CV_8UC1);

	std::vector<uint32_t> random_integers(ASAP::MiscFunctionality::CreateListOfRandomIntegers(tile_coordinates.size()));

	size_t num_to_write = 0 > tile_coordinates.size() ? tile_coordinates.size() : 20;

	cv::Mat tile_image(cv::Mat::zeros(tile_size, tile_size, CV_8UC3));
	
	for (size_t tile = 0; tile < num_to_write; ++tile)
	{
		unsigned char* data = nullptr;
		tiled_image.getRawRegion(tile_coordinates[random_integers[tile]].x * tiled_image.getLevelDownsample(0), tile_coordinates[random_integers[tile]].y * tiled_image.getLevelDownsample(0), tile_size, tile_size, 0, data);

		cv::Mat tile_image = cv::Mat::zeros(tile_size, tile_size, CV_8UC3);
		LevelReading::ArrayToMatrix(data, tile_image, 0, m_is_tiff_);
		delete[] data;

		// get the RGB channels of tile image
		std::vector<cv::Mat> tiled_bgr(5);
		cv::split(tile_image, tiled_bgr);

		cv::Mat& tiled_blue		= tiled_bgr[0];
		cv::Mat& tiled_green	= tiled_bgr[1];
		cv::Mat& tiled_red		= tiled_bgr[2];

		std::vector<uchar> points;
		points.reserve(tiled_blue.cols * tiled_blue.rows);

		for (size_t row = 0; row < tiled_blue.rows; ++row)
		{
			for (size_t col = 0; col < tiled_blue.cols; ++col)
			{
				points.push_back(tiled_blue.at<uchar>(row, col));

			}
		}

		for (size_t row = 0; row < tile_image.rows; ++row)
		{
			for (size_t col = 0; col < tile_image.cols; ++col)
			{
				size_t index							= 256 * 256 * tiled_red.at<unsigned char>(row, col) + 256 * tiled_green.at<unsigned char>(row, col) + tiled_blue.at<unsigned char>(row, col);
				tiled_blue.at<unsigned char>(row, col)	= lut_blue.at<unsigned char>(index, 0);
				tiled_green.at<unsigned char>(row, col) = lut_green.at<unsigned char>(index, 0);
				tiled_red.at<unsigned char>(row, col)	= lut_red.at<unsigned char>(index, 0);
			}
		}

		cv::Mat slide_lut_image;
		cv::merge(tiled_bgr, slide_lut_image);
		
		std::string filename_lut(output_directory.string() + "/" + std::to_string(tile) + "_normalized.tif");
		std::string filename_original(output_directory.string() + "/" + std::to_string(tile) + "_original.tif");

		logging_instance->QueueCommandLineLogging(filename_lut, IO::Logging::NORMAL);

		cv::imwrite(output_directory.string() + "/" + std::to_string(tile) + "blue.tif", tiled_blue);
		cv::imwrite(output_directory.string() + "/" + std::to_string(tile) + "green.tif", tiled_green);
		cv::imwrite(output_directory.string() + "/" + std::to_string(tile) + "red.tif", tiled_red);

		cv::imwrite(filename_lut, slide_lut_image);
		cv::imwrite(filename_original, tile_image);
	}

	logging_instance->QueueCommandLineLogging("Sample images are written...", IO::Logging::NORMAL);
}