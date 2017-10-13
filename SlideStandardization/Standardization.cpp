#include "Standardization.hpp"

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
Standardization::Standardization(std::string log_directory, bool is_multiresolution_image) : is_multiresolution_image(is_multiresolution_image), m_log_file_id_(0)
{
	this->SetLogDirectory(log_directory);
}

void Standardization::CreateNormalizationLUT(
	std::string& input_file,
	std::string& parameters_location,
	std::string& output_directory,
	std::string& debug_directory,
	size_t training_size,
	size_t min_training_size,
	uint32_t tile_size,
	bool is_tiff,
	bool only_generate_parameters,
	bool consider_ink)
{
	IO::Logging::LogHandler* logger_instance(IO::Logging::LogHandler::GetInstance());

	bool RunHough = true;

	//===========================================================================
	//	Reading the image:: Identifying the tiles containing tissue using multiple magnifications
	//===========================================================================
	logger_instance->QueueFileLogging("=============================\n\nReading image...", m_log_file_id_, IO::Logging::NORMAL);

	MultiResolutionImageReader reader;
	MultiResolutionImage* tile_object = reader.open(input_file);

	if (!tile_object)
	{
		throw std::invalid_argument("Unable to open file: " + input_file);
	}

	std::vector<double> spacing			= tile_object->getSpacing();
	if (!spacing.empty())
	{
		if (spacing[0] > 1)
		{
			this->is_multiresolution_image = false;
			spacing.clear();
			logger_instance->QueueFileLogging("Image is static!", m_log_file_id_, IO::Logging::NORMAL);
		}
		else
		{
			logger_instance->QueueFileLogging("Image is multi-resolution", m_log_file_id_, IO::Logging::NORMAL);
		}	
	}

	cv::Mat static_image;
	if (!this->is_multiresolution_image)
	{
		logger_instance->QueueCommandLineLogging("Deconvolving patch image.", IO::Logging::NORMAL);
		static_image = cv::imread(input_file, CV_LOAD_IMAGE_COLOR);
	}

	std::vector<cv::Point> tile_coordinates(GetTileCoordinates_(input_file, spacing, tile_size, is_tiff, this->is_multiresolution_image));

//===========================================================================
//	Performing Pixel Classification
//===========================================================================
	std::string log_text = "Number of available tiles for stain sampling: " + std::to_string(tile_coordinates.size());
	logger_instance->QueueCommandLineLogging(log_text, IO::Logging::NORMAL);
	logger_instance->QueueFileLogging(log_text, m_log_file_id_, IO::Logging::NORMAL);

	PixelClassificationHE pixel_classification_he(consider_ink, m_log_file_id_, debug_directory);
	float hema_percentile	= 0.1; // The higher the value, the more conservative the classifier becomes in picking up blue, so standardization will also be pinkish - breast 0.1
	float eosin_percentile	= 0.2;
	tile_size				= 2048;
	
	SampleInformation sample_training_info(pixel_classification_he.GenerateCxCyDSamples(*tile_object,
																						static_image,
																						tile_coordinates,
																						tile_size,
																						training_size,
																						min_training_size,
																						0,
																						is_tiff,
																						hema_percentile,
																						eosin_percentile,
																						spacing));
	
	logger_instance->QueueCommandLineLogging("sampling done!", IO::Logging::NORMAL);
	logger_instance->QueueFileLogging("=============================\nSampling done!", m_log_file_id_, IO::Logging::NORMAL);

//===========================================================================
//	Generating LUT Raw Matrix
//===========================================================================
	logger_instance->QueueFileLogging("Defining LUT\nLUT HSD", m_log_file_id_, IO::Logging::NORMAL);
	HSD::HSD_Model hsd_lut(CalculateLutRawMat_(), HSD::CHANNEL_SHIFT);

	logger_instance->QueueFileLogging("LUT BG calculation", m_log_file_id_, IO::Logging::NORMAL);
	cv::Mat background_mask(HSD::BackgroundMask::CreateBackgroundMask(hsd_lut, 0.24, 0.22));

//===========================================================================
//	Normalizes the LUT.
//===========================================================================
	cv::Mat normalized_lut(CreateNormalizedImage_(hsd_lut, sample_training_info, training_size, parameters_location, only_generate_parameters));

//===========================================================================
//	Writing LUT image to disk
//===========================================================================
	std::string current_filepath = output_directory;
	std::string output_filepath = current_filepath.substr(0, current_filepath.rfind("_Normalized.tif")) + "_LUT.tif";

	logger_instance->QueueFileLogging("Writing LUT to: " + output_filepath + " (this might take some time).", m_log_file_id_, IO::Logging::NORMAL);
	logger_instance->QueueCommandLineLogging("Writing LUT to: " + output_filepath + " (this might take some time).", IO::Logging::NORMAL);

	cv::imwrite(output_filepath, normalized_lut);

//===========================================================================
//	Write sample images to Harddisk For testing
//===========================================================================
	// Don't remove! usable for looking at samples of standardization
	if (!only_generate_parameters)
	{
		if (this->is_multiresolution_image)
		{
			logger_instance->QueueFileLogging("Writing sample standardized images to: " + debug_directory, m_log_file_id_, IO::Logging::NORMAL);
			std::string filename(core::extractBaseName(input_file));
			WriteSampleNormalizedImagesForTesting_(normalized_lut, *tile_object, tile_size, tile_coordinates, is_tiff, debug_directory, filename);
		}		
		else
		{
			logger_instance->QueueFileLogging("Writing standardized image to: " + debug_directory, m_log_file_id_, IO::Logging::NORMAL);
			WriteSampleNormalizedImagesForTesting_(normalized_lut, static_image, tile_size, is_tiff, debug_directory);
		}
	}
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

SampleInformation Standardization::DownsampleforNbClassifier_(SampleInformation& sample_info, uint32_t downsample, size_t training_size)
{
	SampleInformation sample_info_downsampled
	{
		cv::Mat::zeros(training_size / downsample, 1, CV_32FC1),
		cv::Mat::zeros(training_size / downsample, 1, CV_32FC1),
		cv::Mat::zeros(training_size / downsample, 1, CV_32FC1),
		cv::Mat::zeros(training_size / downsample, 1, CV_32FC1)
	};

	for (size_t downsampled_pixel = 0; downsampled_pixel < sample_info.class_data.rows / downsample; ++downsampled_pixel)
	{
		sample_info_downsampled.training_data_c_x.at<float>(downsampled_pixel, 0)		= sample_info.training_data_c_x.at<float>(downsampled_pixel * downsample, 0);
		sample_info_downsampled.training_data_c_y.at<float>(downsampled_pixel, 0)		= sample_info.training_data_c_y.at<float>(downsampled_pixel * downsample, 0);
		sample_info_downsampled.training_data_density.at<float>(downsampled_pixel, 0)	= sample_info.training_data_density.at<float>(downsampled_pixel * downsample, 0);
		sample_info_downsampled.class_data.at<float>(downsampled_pixel, 0)				= sample_info.class_data.at<float>(downsampled_pixel * downsample, 0);
	}

	return sample_info_downsampled;
}

void Standardization::WriteNormalizedWSI(std::string& slide_directory, cv::Mat& normalized_lut, uint32_t tile_size, bool is_tiff, std::string& output_filepath)
{
	std::vector<cv::Mat> lut_bgr(3);
	cv::split(normalized_lut, lut_bgr);

	cv::Mat lut_blue	= lut_bgr[0];
	cv::Mat lut_green	= lut_bgr[1]; 
	cv::Mat lut_red		= lut_bgr[2]; 

	lut_blue.convertTo(lut_blue, CV_8UC1);
	lut_green.convertTo(lut_green, CV_8UC1);
	lut_red.convertTo(lut_red, CV_8UC1);

	MultiResolutionImageWriter image_writer;
	image_writer.openFile(output_filepath);
	image_writer.setTileSize(tile_size);
	image_writer.setCompression(pathology::LZW);
	image_writer.setDataType(pathology::UChar);
	image_writer.setColorType(pathology::RGB);
	image_writer.writeImageInformation(tile_size * 4, tile_size * 4);

	MultiResolutionImageReader reader;
	MultiResolutionImage* tiled_object	= reader.open(slide_directory);
	std::vector<size_t> dimensions		= tiled_object->getLevelDimensions(0);
	image_writer.writeImageInformation(dimensions[0], dimensions[1]);
	
	uint64_t y_amount_of_tiles		= std::ceil((float)dimensions[0] / (float)tile_size);
	uint64_t x_amount_of_tiles		= std::ceil((float)dimensions[1] / (float)tile_size);
	uint64_t total_amount_of_tiles	= x_amount_of_tiles * y_amount_of_tiles;

	std::vector<uint64_t> x_values, y_values;

	for (uint64_t x_tile = 0; x_tile < x_amount_of_tiles; ++x_tile)
	{
		for (uint64_t y_tile = 0; y_tile < y_amount_of_tiles; ++y_tile)
		{
			y_values.push_back(tile_size*y_tile);
			x_values.push_back(tile_size*x_tile);
		}
	}

	for (uint64_t tile = 0; tile < total_amount_of_tiles; ++tile)
	{   
		std::vector<unsigned char> data(tile_size * tile_size * 4);
		unsigned char* data_ptr(&data[0]);
		tiled_object->getRawRegion(y_values[tile] * tiled_object->getLevelDownsample(0), x_values[tile] * tiled_object->getLevelDownsample(0), tile_size, tile_size, 0, data_ptr);
		
		std::vector<unsigned char> modified_data(tile_size * tile_size * 3);
		unsigned char* modified_data_ptr(&data[0]);

		size_t data_index			= 0;
		size_t modified_data_index	= 0;
		for (size_t pixel = 0; pixel < tile_size * tile_size; ++pixel)
		{
			size_t index = 256 * 256 * data[data_index] + 256 * data[data_index + 1] + data[data_index + 2];
			modified_data[++modified_data_index] = lut_red.at<unsigned char>(index, 0);
			modified_data[++modified_data_index] = lut_green.at<unsigned char>(index, 0);
			modified_data[++modified_data_index] = lut_blue.at<unsigned char>(index, 0);
			data_index = data_index + 3;
		}
		image_writer.writeBaseImagePart((void*)modified_data_ptr);
	}
	image_writer.finishImage();
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

cv::Mat Standardization::CreateNormalizedImage_(HSD::HSD_Model& hsd_lut, SampleInformation& sample_info, size_t training_size, std::string& parameters_filepath, bool only_generate_parameters)
{
	IO::Logging::LogHandler* logger_instance(IO::Logging::LogHandler::GetInstance());

	//===========================================================================
	//	Transforming Cx and Cy distributions - Initialization 1
	//  Extracting all the parameters needed for applying the transformation
	//===========================================================================
	logger_instance->QueueFileLogging("Defining variables for transformation...", m_log_file_id_, IO::Logging::NORMAL);

	ClassAnnotatedCxCy train_data(TransformCxCyDensity::ClassCxCyGenerator(sample_info.class_data, sample_info.training_data_c_x, sample_info.training_data_c_y));
	
	// Rotates the cx_cy matrice per class and stores the parameters used.
	cv::Mat cx_cy_hema_rotated;
	cv::Mat cx_cy_eosin_rotated;
	cv::Mat cx_cy_background_Rotated;

	MatrixRotationParameters hema_rotation_info(TransformCxCyDensity::RotateCxCy(train_data.cx_cy_merged, cx_cy_hema_rotated, train_data.hema_cx_cy));
	MatrixRotationParameters eosin_rotation_info(TransformCxCyDensity::RotateCxCy(train_data.cx_cy_merged, cx_cy_eosin_rotated, train_data.eosin_cx_cy));
	MatrixRotationParameters background_rotation_info(TransformCxCyDensity::RotateCxCy(train_data.cx_cy_merged, cx_cy_background_Rotated, train_data.background_cx_cy));

	ClassPixelIndices class_pixel_indices(TransformCxCyDensity::GetClassIndices(sample_info.class_data));

	// Calculates the scale parameters per class.
	cv::Mat hema_scale_parameters(TransformCxCyDensity::CalculateScaleParameters(class_pixel_indices.hema_indices, cx_cy_hema_rotated));
	cv::Mat eosin_scale_parameters(TransformCxCyDensity::CalculateScaleParameters(class_pixel_indices.eosin_indices, cx_cy_eosin_rotated));
	cv::Mat background_scale_parameters(TransformCxCyDensity::CalculateScaleParameters(class_pixel_indices.background_indices, cx_cy_background_Rotated));

	ClassDensityRanges class_density_ranges(TransformCxCyDensity::GetDensityRanges(sample_info.class_data, sample_info.training_data_density, class_pixel_indices));

	logger_instance->QueueFileLogging("Finished computing tranformation parameters for the current image", m_log_file_id_, IO::Logging::NORMAL);
	//===========================================================================
	//	Defining Template Parameters
	//===========================================================================

	TransformationParameters calculated_transform_parameters{ hema_rotation_info, eosin_rotation_info, background_rotation_info, hema_scale_parameters, eosin_scale_parameters, class_density_ranges };
	TransformationParameters lut_transform_parameters(calculated_transform_parameters);

	HandleParameterization(calculated_transform_parameters, lut_transform_parameters, parameters_filepath);

	if (only_generate_parameters)
	{
		return cv::Mat();
	}

	//===========================================================================
	//	Generating the weights for each class
	//===========================================================================
	// Downsample the number of samples for NB classifier
	uint32_t downsample = 20;
	if (training_size > 10000000 && training_size < 20000000)
	{
		downsample = 30;
	}
	else if (training_size >= 20000000 && training_size < 30000000)
	{
		downsample = 40;
	}
	else if (training_size >= 30000000)
	{
		downsample = 50;
	}

	logger_instance->QueueFileLogging("Down sampling the data for constructing NB classifier", m_log_file_id_, IO::Logging::NORMAL);

	SampleInformation sample_info_downsampled(DownsampleforNbClassifier_(sample_info, downsample, training_size));

	logger_instance->QueueFileLogging("Generating weights with NB classifier", m_log_file_id_, IO::Logging::NORMAL);
	logger_instance->QueueCommandLineLogging("Generating the weights, Setting dataset of size " + std::to_string(sample_info_downsampled.class_data.rows * sample_info_downsampled.class_data.cols),
		IO::Logging::NORMAL);

	cv::Ptr<cv::ml::NormalBayesClassifier> classifier(CxCyWeights::CreateNaiveBayesClassifier(	sample_info_downsampled.training_data_c_x,	
																								sample_info_downsampled.training_data_c_y, 
																								sample_info_downsampled.training_data_density, 
																								sample_info_downsampled.class_data));
	logger_instance->QueueCommandLineLogging("Training Naive Bayes Classifier fininshed...", IO::Logging::NORMAL);

	logger_instance->QueueCommandLineLogging("Generating posteriors (This will take some time...)", IO::Logging::NORMAL);
	CxCyWeights::Weights weights(CxCyWeights::GenerateWeights(hsd_lut.c_x, hsd_lut.c_y, hsd_lut.density, classifier));
	logger_instance->QueueCommandLineLogging("All weights created...", IO::Logging::NORMAL);
	logger_instance->QueueFileLogging("Weights generated", m_log_file_id_, IO::Logging::NORMAL);

	//===========================================================================
	//	Transforming Cx and Cy distributions - Initialization
	//===========================================================================
	logger_instance->QueueCommandLineLogging("Transformation started...", IO::Logging::NORMAL);
	logger_instance->QueueFileLogging("Transformation started...", m_log_file_id_, IO::Logging::NORMAL);

	std::vector<cv::Mat> lut_transformation_results(InitializeTransformation_(hsd_lut, train_data.cx_cy_merged, calculated_transform_parameters, lut_transform_parameters, class_pixel_indices));

	//===========================================================================
	//	Generating the weights for each class
	//===========================================================================
	logger_instance->QueueFileLogging("Applying weights...", m_log_file_id_, IO::Logging::NORMAL);
	cv::Mat cx_cy_normalized(CxCyWeights::ApplyWeights(lut_transformation_results[0], lut_transformation_results[0], lut_transformation_results[1], weights));

	//===========================================================================
	//	Density scaling
	//===========================================================================
	//TransCxCyD.densityNormalization(trainDataAllD, HSD_LUT.Density, BG_Obj_LUT);
	logger_instance->QueueFileLogging("Density transformation...", m_log_file_id_, IO::Logging::NORMAL);
	cv::Mat density_scaling(TransformCxCyDensity::DensityNormalizationThreeScales(calculated_transform_parameters.class_density_ranges, lut_transform_parameters.class_density_ranges, hsd_lut.density, weights));

	//===========================================================================
	//	HSD reverse
	//===========================================================================
	logger_instance->QueueFileLogging("HSD reverse...", m_log_file_id_, IO::Logging::NORMAL);
	cv::Mat nornmalized_image_rgb;
	HSD::CxCyToRGB(cx_cy_normalized, nornmalized_image_rgb, density_scaling);

	return nornmalized_image_rgb;
}

std::vector<cv::Point> Standardization::GetTileCoordinates_(std::string& slide_directory, std::vector<double>& spacing, uint32_t tile_size, bool is_tiff, bool is_multi_resolution)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	int min_level = 0;
	std::vector<cv::Point> tile_coordinates;
	if (is_multi_resolution)
	{
		MultiResolutionImageReader reader;
		MultiResolutionImage* ReadTileObject = reader.open(slide_directory);
		if (spacing.size() > 0)
		{
			spacing = ReadTileObject->getSpacing();
			if (spacing[0] < 0.2)
			{
				spacing[0] = spacing[0] * 2;
				min_level = 1;
			}
			logging_instance->QueueFileLogging("Pixel spacing = " + std::to_string(spacing[0]), m_log_file_id_, IO::Logging::NORMAL);
		}
		else
		{
			logging_instance->QueueCommandLineLogging("The image does not have spacing information. Continuing with the default 0.24.", IO::Logging::NORMAL);
			spacing.push_back(0.243);
			logging_instance->QueueFileLogging("The image does not have spacing information. Continuing with the default 0.24. Pixel spacing set to default = " + std::to_string(spacing[0]), m_log_file_id_, IO::Logging::NORMAL);
		}

		unsigned char number_of_levels = ReadTileObject->getNumberOfLevels();
		if (number_of_levels > 5)
		{
			number_of_levels = 5;
		}

		logging_instance->QueueFileLogging("Number of levels available = " + std::to_string(number_of_levels), m_log_file_id_, IO::Logging::NORMAL);

		logging_instance->QueueCommandLineLogging("detecting tissue regions...", IO::Logging::NORMAL);
		logging_instance->QueueFileLogging("Detecting tissue", m_log_file_id_, IO::Logging::NORMAL);

		// Attempts to acquire the tile coordinates for the lowest level / highest magnification.
		std::vector<size_t> dimensions = ReadTileObject->getLevelDimensions(number_of_levels - 1);
		uint32_t skip_factor = 1;
		float background_tissue_threshold = 0.9;
		uint32_t level_scale_difference = 1;

		if (number_of_levels > 1)
		{
			logging_instance->QueueCommandLineLogging("Analyzing level: " + std::to_string(number_of_levels - 1), IO::Logging::NORMAL);

			std::vector<unsigned long long> next_level_dimensions = ReadTileObject->getLevelDimensions(number_of_levels - 2);
			level_scale_difference = std::pow(std::round(next_level_dimensions[0] / next_level_dimensions[0]), 2);

			// Loops through each level, acquiring coordinates for each and reusing them to calculate the set of coordinates for a higher magnification.
			tile_coordinates.swap(LevelReading::ReadLevelTiles(*ReadTileObject, dimensions[0], dimensions[1], tile_size, is_tiff, number_of_levels - 1, background_tissue_threshold, skip_factor));
			for (char level_number = number_of_levels - 2; level_number >= 0; --level_number)
			{
				if (level_number != 0)
				{
					skip_factor = 2;

					std::vector<size_t> dimensions_level_low = ReadTileObject->getLevelDimensions(level_number);
					std::vector<size_t> dimensions_level_high = ReadTileObject->getLevelDimensions(level_number - 1);
					level_scale_difference = std::pow(std::floor(dimensions_level_high[0] / dimensions_level_low[0]), 2);
				}

				std::string log_text = "Analyzing level: " + std::to_string(level_number) + " - Tiles containing tissue: " + std::to_string(tile_coordinates.size());
				logging_instance->QueueCommandLineLogging(log_text, IO::Logging::NORMAL);
				logging_instance->QueueFileLogging(log_text, m_log_file_id_, IO::Logging::NORMAL);

				background_tissue_threshold -= 0.1;
				tile_coordinates.swap(LevelReading::ReadLevelTiles(*ReadTileObject, tile_coordinates, tile_size, is_tiff, level_number, background_tissue_threshold, skip_factor, level_scale_difference));
			}
		}
		else
		{
			tile_coordinates.swap(LevelReading::ReadLevelTiles(*ReadTileObject, dimensions[0], dimensions[1], tile_size, is_tiff, number_of_levels - 1, 0.9, skip_factor));
		}
	}
	else
	{
		tile_coordinates.push_back(cv::Point(0, 0));
		spacing.push_back(0.243);
	}

	return tile_coordinates;
}

void Standardization::HandleParameterization(TransformationParameters& calc_params, TransformationParameters& lut_params, std::string& parameters_filepath)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	if (!parameters_filepath.empty())
	{
		logging_instance->QueueFileLogging("Loading Template parameters...", m_log_file_id_, IO::Logging::NORMAL);
		std::ifstream csv_input_stream;
		csv_input_stream.open(parameters_filepath);
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
	else
	{
		logging_instance->QueueFileLogging("Saving Template parameters...", m_log_file_id_, IO::Logging::NORMAL);
		std::ofstream csv_output_stream;
		csv_output_stream.open(parameters_filepath);
		if (csv_output_stream)
		{
			PrintParameters_(csv_output_stream, calc_params, true);
			csv_output_stream.close();

			logging_instance->QueueCommandLineLogging("Done", IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Template Parameters written to: " + parameters_filepath, m_log_file_id_, IO::Logging::NORMAL);
		}
		else
		{
			logging_instance->QueueCommandLineLogging("Could not read/write template CSV file!", IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Could not read/write template CSV file!", m_log_file_id_, IO::Logging::NORMAL);
		}
	}
}

std::vector<cv::Mat> Standardization::InitializeTransformation_(
	HSD::HSD_Model& hsd_lut,
	cv::Mat& cx_cy_train_data,
	TransformationParameters& params,
	TransformationParameters& transform_params,
	ClassPixelIndices& class_pixel_indices)
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

void Standardization::PrintParameters_(std::ofstream& output_stream, TransformationParameters& transform_param,	bool write_csv)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	logging_instance->QueueCommandLineLogging("\nHematoxylin CxCy: "	+ std::to_string(transform_param.hema_rotation_params.x_median)			+ ", " + std::to_string(transform_param.hema_rotation_params.y_median)		+ "\n", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Eosin CxCy: "			+ std::to_string(transform_param.eosin_rotation_params.x_median)		+ ", " + std::to_string(transform_param.eosin_rotation_params.y_median)		+ "\n", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Background CxCy: "		+ std::to_string(transform_param.background_rotation_params.x_median)	+ ", " + std::to_string(transform_param.background_rotation_params.y_median)	+ "\n", IO::Logging::NORMAL);
	

	cv::Mat& hema_scale_param(transform_param.hema_scale_params);
	cv::Mat& eosin_scale_param(transform_param.eosin_scale_params);

	logging_instance->QueueCommandLineLogging("Hema Cx values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(hema_scale_param.at<float>(0, 0)) + std::to_string(hema_scale_param.at<float>(1, 0)) + std::to_string(hema_scale_param.at<float>(2, 0)) +
		std::to_string(hema_scale_param.at<float>(3, 0)) + std::to_string(hema_scale_param.at<float>(4, 0)) + std::to_string(hema_scale_param.at<float>(5, 0)) + std::to_string(hema_scale_param.at<float>(6, 0)), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\nHema Cy values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(hema_scale_param.at<float>(0, 1)) + std::to_string(hema_scale_param.at<float>(1, 1)) + std::to_string(hema_scale_param.at<float>(2, 1)) +
		std::to_string(hema_scale_param.at<float>(3, 1)) + std::to_string(hema_scale_param.at<float>(4, 1)) + std::to_string(hema_scale_param.at<float>(5, 1)) + std::to_string(hema_scale_param.at<float>(6, 1)), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\nEosin Cx values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(eosin_scale_param.at<float>(0, 0)) + std::to_string(eosin_scale_param.at<float>(1, 0)) + std::to_string(eosin_scale_param.at<float>(2, 0)) +
		std::to_string(eosin_scale_param.at<float>(3, 0)) + std::to_string(eosin_scale_param.at<float>(4, 0)) + std::to_string(eosin_scale_param.at<float>(5, 0)) + std::to_string(eosin_scale_param.at<float>(6, 0)), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\nEosin Cy values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(eosin_scale_param.at<float>(0, 1)) + std::to_string(eosin_scale_param.at<float>(1, 1)) + std::to_string(eosin_scale_param.at<float>(2, 1)) +
		std::to_string(eosin_scale_param.at<float>(3, 1)) + std::to_string(eosin_scale_param.at<float>(4, 1)) + std::to_string(eosin_scale_param.at<float>(5, 1)) + std::to_string(eosin_scale_param.at<float>(6, 1)), IO::Logging::NORMAL);


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

void Standardization::WriteSampleNormalizedImagesForTesting_(cv::Mat& normalized_lut, cv::Mat& tile_image, uint32_t tile_size, bool is_tiff, std::string& output_directory)
{
	std::vector<cv::Mat> rgb;
	cv::split(normalized_lut, rgb);

	cv::Mat& blue_lut = rgb[0];
	cv::Mat& green_lut = rgb[1];
	cv::Mat& red_lut = rgb[2];

	blue_lut.convertTo(blue_lut, CV_8UC1);
	green_lut.convertTo(green_lut, CV_8UC1);
	red_lut.convertTo(red_lut, CV_8UC1);

	// get the BGR channels of tile image
	std::vector<cv::Mat> channels_in_tile(3);
	cv::split(tile_image, channels_in_tile);
	cv::Mat blue = channels_in_tile[0];
	cv::Mat green = channels_in_tile[1];
	cv::Mat red = channels_in_tile[2];

	for (size_t row = 0; row < tile_image.rows; ++row)
	{
		for (size_t col = 0; col < tile_image.cols; ++col)
		{
			size_t index = 256 * 256 * red.at<unsigned char>(row, col) + 256 * green.at<unsigned char>(row, col) + blue.at<unsigned char>(row, col);
			if (row != 0 && col != 0)
			{
				blue.at<unsigned char>(row, col) = blue_lut.at<unsigned char>(index, 0);
				green.at<unsigned char>(row, col) = green_lut.at<unsigned char>(index, 0);
				red.at<unsigned char>(row, col) = red_lut.at<unsigned char>(index, 0);
			}
		}
	}

	cv::Mat lut_slide_image;
	cv::merge(channels_in_tile, lut_slide_image);

	cv::imwrite(output_directory, lut_slide_image);

	IO::Logging::LogHandler* logger_instance(IO::Logging::LogHandler::GetInstance());
	logger_instance->QueueCommandLineLogging("Normalized image is written...", IO::Logging::NORMAL);
}

void Standardization::WriteSampleNormalizedImagesForTesting_(
	cv::Mat& lut_image,
	MultiResolutionImage& tiled_image,
	uint32_t tile_size,
	std::vector<cv::Point>& tile_coordinates,
	bool is_tiff,
	std::string& debug_dir,
	std::string& filename)
{
	boost::filesystem::path current_dir(debug_dir + "/DebugData/" + filename + "norm");
	boost::filesystem::create_directory(current_dir);

	IO::Logging::LogHandler* logger_instance(IO::Logging::LogHandler::GetInstance());

	logger_instance->QueueCommandLineLogging("Writing sample standardized images in: " + current_dir.string(), IO::Logging::NORMAL);

	std::vector<cv::Mat> lut_bgr(3);
	cv::split(lut_image, lut_bgr);

	cv::Mat& lut_blue = lut_bgr[0];
	cv::Mat& lut_green = lut_bgr[1];
	cv::Mat& lut_red = lut_bgr[2];

	lut_red.convertTo(lut_red, CV_8UC1);
	lut_green.convertTo(lut_green, CV_8UC1);
	lut_blue.convertTo(lut_blue, CV_8UC1);

	std::vector<uint32_t> random_integers(ASAP::MiscFunctionality::CreateListOfRandomIntegers(tile_coordinates.size()));

	size_t num_to_write = 0 > tile_coordinates.size() ? tile_coordinates.size() : 20;

	cv::Mat tile_image(cv::Mat::zeros(tile_size, tile_size, CV_8UC3));
	std::vector<unsigned char> data(tile_size * tile_size * 4);
	unsigned char* data_ptr(&data[0]);
	for (size_t tile = 0; tile < num_to_write; ++tile)
	{
		tiled_image.getRawRegion(tile_coordinates[random_integers[tile]].x * tiled_image.getLevelDownsample(0), tile_coordinates[random_integers[tile]].y * tiled_image.getLevelDownsample(0), tile_size, tile_size, 0, data_ptr);

		cv::Mat tile_image = cv::Mat::zeros(tile_size, tile_size, CV_8UC3);
		LevelReading::ArrayToMatrix(data_ptr, tile_image, 0, is_tiff);

		// get the RGB channels of tile image
		std::vector<cv::Mat> tiled_bgr(3);
		cv::split(tile_image, tiled_bgr);

		cv::Mat& tiled_red = tiled_bgr[2];
		cv::Mat& tiled_green = tiled_bgr[1];
		cv::Mat& tiled_blue = tiled_bgr[0];

		for (size_t row = 1; row < tile_image.rows; ++row)
		{
			for (size_t col = 1; col < tile_image.cols; ++col)
			{
				size_t index = 256 * 256 * tiled_red.at<unsigned char>(row, col) + 256 * tiled_green.at<unsigned char>(row, col) + tiled_blue.at<unsigned char>(row, col);
				tiled_blue.at<unsigned char>(row, col) = lut_blue.at<unsigned char>(index, 0);
				tiled_green.at<unsigned char>(row, col) = lut_green.at<unsigned char>(index, 0);
				tiled_red.at<unsigned char>(row, col) = lut_red.at<unsigned char>(index, 0);
			}
		}

		cv::Mat slide_lut_image;
		cv::merge(tiled_bgr, slide_lut_image);

		std::string filename_lut(current_dir.string() + "/" + std::to_string(tile) + "_Normalized.tif");
		std::string filename_original(current_dir.string() + "/" + std::to_string(tile) + "_Original.tif");

		logger_instance->QueueCommandLineLogging(filename_lut, IO::Logging::NORMAL);

		cv::imwrite(filename_lut, slide_lut_image);
		cv::imwrite(filename_original, tile_image);
	}

	logger_instance->QueueCommandLineLogging("Sample images are written...", IO::Logging::NORMAL);
}