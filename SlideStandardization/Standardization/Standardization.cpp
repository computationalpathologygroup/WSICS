#include "Standardization.h"

#include <boost/filesystem.hpp>
#include <stdexcept>
#include <core/filetools.h>

#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImageWriter.h"
#include "multiresolutionimageinterface/OpenSlideImageFactory.h"

#include "CxCyWeights.h"
#include "NormalizedLutCreation.h"
#include "../HSD/BackgroundMask.h"
#include "../HSD/Transformations.h"
#include "../IO/Logging/LogHandler.h"
#include "../Misc/LevelReading.h"
#include "../Misc/MiscFunctionality.h"

// TODO: Refactor and restructure into smaller chunks.

Standardization::Standardization(std::string log_directory,	const boost::filesystem::path& template_file)
	: m_log_file_id_(0), m_template_file_(template_file), m_debug_directory_(), m_parameters_(GetStandardParameters()), m_is_multiresolution_image_(false)
{
	this->SetLogDirectory(log_directory);
}

Standardization::Standardization(std::string log_directory, const boost::filesystem::path& template_file, const StandardizationParameters& parameters)
	: m_log_file_id_(0), m_template_file_(template_file), m_debug_directory_(), m_parameters_(parameters), m_is_multiresolution_image_(false)
{
	this->SetLogDirectory(log_directory);
}

StandardizationParameters Standardization::GetStandardParameters(void)
{
	return { -1, 200000, 20000000, 0.1f, 0.2f, false };
}

void Standardization::Normalize(
	const boost::filesystem::path& input_file,
	const boost::filesystem::path& output_file,
	const boost::filesystem::path& template_output,
	const boost::filesystem::path& debug_directory)
{
	//===========================================================================
	//	Sets several execution variables.
	//===========================================================================

	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
	m_debug_directory_	= debug_directory;
	
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

	HSD::HSD_Model lut_hsd(CalculateLutRawMat_(), HSD::BGR);

	logging_instance->QueueFileLogging("LUT BG calculation", m_log_file_id_, IO::Logging::NORMAL);
	cv::Mat background_mask(HSD::BackgroundMask::CreateBackgroundMask(lut_hsd, 0.24, 0.22));

	//===========================================================================
	//	Normalizes the LUT.
	//===========================================================================
	cv::Mat normalized_lut(NormalizedLutCreation::Create(!output_file.empty(), m_template_file_, template_output, lut_hsd, training_samples, m_parameters_.max_training_size, m_log_file_id_));

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
				WriteSampleNormalizedImagesForTesting_(boost::filesystem::path(m_debug_directory_.string()), normalized_lut, *tiled_image, tile_coordinates, tile_size);
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

	PixelClassificationHE pixel_classification_he(m_parameters_.consider_ink, m_log_file_id_, m_debug_directory_.string());
	tile_size = 2048;

	return pixel_classification_he.GenerateCxCyDSamples(
		tiled_image,
		static_image,
		m_parameters_,
		tile_coordinates,
		spacing,
		tile_size,
		min_level,
		m_is_multiresolution_image_);
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
		tile_coordinates.swap(LevelReading::ReadLevelTiles(tiled_image, dimensions[0], dimensions[1], tile_size, number_of_levels - 1, skip_factor, background_tissue_threshold));
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
			tile_coordinates.swap(LevelReading::ReadLevelTiles(tiled_image, tile_coordinates, tile_size, level_number, skip_factor, level_scale_difference, background_tissue_threshold));
		}
	}
	else
	{
		tile_coordinates.swap(LevelReading::ReadLevelTiles(tiled_image, dimensions[0], dimensions[1], tile_size, number_of_levels - 1, 0.9, skip_factor));
	}

	return tile_coordinates;
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

	MultiResolutionImageReader reader;
	MultiResolutionImage* tiled_image	= reader.open(input_file.string());
	std::vector<size_t> dimensions		= tiled_image->getLevelDimensions(0);

	MultiResolutionImageWriter image_writer;
	image_writer.openFile(output_file.string());
	image_writer.setTileSize(tile_size);
	image_writer.setCompression(pathology::LZW);
	image_writer.setDataType(pathology::UChar);
	image_writer.setColorType(pathology::RGB);
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

	std::vector<size_t> random_integers(ASAP::MiscFunctionality::CreateListOfRandomIntegers(tile_coordinates.size()));

	size_t num_to_write = 0 > tile_coordinates.size() ? tile_coordinates.size() : 20;
	cv::Mat tile_image(cv::Mat::zeros(tile_size, tile_size, CV_8UC3));
	for (size_t tile = 0; tile < num_to_write; ++tile)
	{
		unsigned char* data = nullptr;
		tiled_image.getRawRegion(tile_coordinates[random_integers[tile]].x * tiled_image.getLevelDownsample(0), tile_coordinates[random_integers[tile]].y * tiled_image.getLevelDownsample(0), tile_size, tile_size, 0, data);

		cv::Mat tile_image = cv::Mat::zeros(tile_size, tile_size, CV_8UC3);
		LevelReading::ArrayToMatrix(data, tile_image, 0);
		delete[] data;

		// get the RGB channels of tile image
		std::vector<cv::Mat> tiled_bgr(3);
		cv::split(tile_image, tiled_bgr);

		cv::Mat& tiled_blue		= tiled_bgr[0];
		cv::Mat& tiled_green	= tiled_bgr[1];
		cv::Mat& tiled_red		= tiled_bgr[2];

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
		std::string filename_lut(output_directory.string() + "/" + "tile_" + std::to_string(random_integers[tile]) + "_normalized.tif");
		logging_instance->QueueCommandLineLogging(filename_lut, IO::Logging::NORMAL);
		cv::imwrite(filename_lut, slide_lut_image);
	}

	logging_instance->QueueCommandLineLogging("Sample images are written...", IO::Logging::NORMAL);
}