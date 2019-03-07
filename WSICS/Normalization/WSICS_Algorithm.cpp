#include "WSICS_Algorithm.h"

#include <multiresolutionimageinterface/MultiResolutionImageReader.h>
#include <multiresolutionimageinterface/MultiResolutionImageFactory.h>
#include <boost/filesystem.hpp>
#include <stdexcept>
#include <core/filetools.h>

#include "CxCyWeights.h"
#include "NormalizedLutCreation.h"
#include "NormalizedOutput.h"
#include "../HSD/BackgroundMask.h"
#include "../HSD/Transformations.h"
#include "../IO/Logging/LogHandler.h"
#include "../Misc/LevelReading.h"
#include "../Misc/Random.h"

// TODO: Refactor and restructure into smaller chunks.

namespace WSICS::Normalization
{
	WSICS_Algorithm::WSICS_Algorithm(std::string log_directory, const boost::filesystem::path& template_file)
		: m_log_file_id_(0), m_template_file_(template_file), m_debug_directory_(), m_parameters_(GetStandardParameters()), m_is_multiresolution_image_(false)
	{
		this->SetLogDirectory(log_directory);
	}

	WSICS_Algorithm::WSICS_Algorithm(std::string log_directory, const boost::filesystem::path& template_file, const WSICS_Parameters& parameters)
		: m_log_file_id_(0), m_template_file_(template_file), m_debug_directory_(), m_parameters_(parameters), m_is_multiresolution_image_(false)
	{
		this->SetLogDirectory(log_directory);
	}

	WSICS_Parameters WSICS_Algorithm::GetStandardParameters(void)
	{
		return { -1, 200000, 20000000, 2000, 0.1f, 0.2f, 0.9f, false };
	}

	void WSICS_Algorithm::Normalize(
		const boost::filesystem::path& input_file,
		const boost::filesystem::path& image_output_file,
		const boost::filesystem::path& lut_output_file,
		const boost::filesystem::path& template_output_file,
		const boost::filesystem::path& debug_directory)
	{
		//===========================================================================
		//	Sets several execution variables.
		//===========================================================================

		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
		m_debug_directory_ = debug_directory;

		//===========================================================================
		//	Reading the image:: Identifying the tiles containing tissue using multiple magnifications
		//===========================================================================
		logging_instance->QueueFileLogging("=============================\n\nReading image...", m_log_file_id_, IO::Logging::NORMAL);

		MultiResolutionImageReader reader;
		MultiResolutionImage* tiled_image = reader.open(input_file.string());

		if (!tiled_image)
		{
			std::set<std::string> extensions(MultiResolutionImageFactory::getAllSupportedExtensions());
			std::string not_supported = "\n";
			if (extensions.find(input_file.extension().string().substr(1)) == extensions.end())
			{
				not_supported += input_file.extension().string().substr(1) + " files aren't supported by this binary";
			}

			throw std::invalid_argument("Unable to open file: " + input_file.string() + not_supported);
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
			spacing[0] *= 2;
			min_level = 1;
		}

		logging_instance->QueueFileLogging("Pixel spacing = " + std::to_string(spacing[0]), m_log_file_id_, IO::Logging::NORMAL);

		uint32_t tile_size = 512;
		cv::Mat static_image;
		std::vector<cv::Point> tile_coordinates;
		if (m_is_multiresolution_image_)
		{
			tile_coordinates = std::move(GetTileCoordinates_(*tiled_image, spacing, tile_size, min_level));
		}
		else
		{
			logging_instance->QueueCommandLineLogging("Deconvolving patch image.", IO::Logging::NORMAL);
			static_image = cv::imread(input_file.string(), cv::IMREAD_COLOR);
			tile_coordinates.push_back({ 0, 0 });
		}

		if (tile_coordinates.size() == 0)
		{
			throw std::runtime_error("Unable to acquire tiles with tissue. (Try changing the background threshold parameter)");
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
		cv::Mat normalized_lut(NormalizedLutCreation::Create(!image_output_file.empty() || !lut_output_file.empty(), m_template_file_, template_output_file, lut_hsd, training_samples, m_parameters_.max_training_size, m_log_file_id_));

		if (!lut_output_file.empty())
		{
			logging_instance->QueueFileLogging("Writing LUT to: " + lut_output_file.string() + " (this might take some time).", m_log_file_id_, IO::Logging::NORMAL);
			logging_instance->QueueCommandLineLogging("Writing LUT to: " + lut_output_file.string() + " (this might take some time).", IO::Logging::NORMAL);
			cv::imwrite(lut_output_file.string(), normalized_lut);
		}

		//===========================================================================
		//	Writing LUT image to disk
		//===========================================================================
		if (!image_output_file.empty())
		{
			logging_instance->QueueFileLogging("Writing the standardized WSI in progress...", m_log_file_id_, IO::Logging::NORMAL);
			logging_instance->QueueCommandLineLogging("Writing the standardized WSI in progress...", IO::Logging::NORMAL);

			if (m_is_multiresolution_image_)
			{
				WriteNormalizedWSI(input_file, image_output_file, normalized_lut, tile_size);
			}
			else
			{
				WriteNormalizedWSI(static_image, image_output_file, normalized_lut);
			}
			logging_instance->QueueFileLogging("Finished writing the image.", m_log_file_id_, IO::Logging::NORMAL);
			logging_instance->QueueCommandLineLogging("Finished writing the image.", IO::Logging::NORMAL);
		}

		//===========================================================================
		//	Write sample images to Harddisk For testing
		//===========================================================================
		// Don't remove! usable for looking at samples of standardization
		if (!m_debug_directory_.empty() && logging_instance->GetOutputLevel() == IO::Logging::DEBUG)
		{
			logging_instance->QueueFileLogging("Writing sample standardized images to: " + m_debug_directory_.string(), m_log_file_id_, IO::Logging::NORMAL);

			if (m_is_multiresolution_image_)
			{
				WriteNormalizedSamples(boost::filesystem::path(m_debug_directory_.string()), normalized_lut, *tiled_image, tile_coordinates, tile_size);
			}
			else
			{
				boost::filesystem::path output_filepath(m_debug_directory_.string() + "/" + input_file.stem().string() + ".tif");
				WriteNormalizedSample(output_filepath.string(), normalized_lut, static_image, tile_size);
			}
		}

		//===========================================================================
		//	Cleans execution variables
		//===========================================================================
		m_debug_directory_ = "";
		delete tiled_image;
	}

	void WSICS_Algorithm::SetLogDirectory(std::string& filepath)
	{
		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

		// Closes the current log file, if present.
		if (m_log_file_id_ > 0)
		{
			logging_instance->CloseFile(m_log_file_id_);
		}

		m_log_file_id_ = logging_instance->OpenFile(filepath, false);
	}

	cv::Mat WSICS_Algorithm::CalculateLutRawMat_(void)
	{
		cv::Mat raw_lut(cv::Mat::zeros(256 * 256 * 256, 1, CV_8UC3));//16387064
		int counterTest = 0;
		for (int i1 = 0; i1<256; ++i1)
		{
			for (int i2 = 0; i2<256; ++i2)
			{
				for (int i3 = 0; i3<256; ++i3)
				{
					raw_lut.at<cv::Vec3b>(counterTest, 0) = cv::Vec3b(i3, i2, i1);
					++counterTest;
				}
			}
		}
		return raw_lut;
	}

	TrainingSampleInformation WSICS_Algorithm::CollectTrainingSamples_(
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

	std::pair<bool, std::vector<double>> WSICS_Algorithm::GetResolutionTypeAndSpacing(MultiResolutionImage& tiled_image)
	{
		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
		std::pair<bool, std::vector<double>> resolution_and_spacing((tiled_image.getNumberOfLevels() > 1), tiled_image.getSpacing());

		if (!resolution_and_spacing.second.empty())
		{
			if (resolution_and_spacing.second[0] > 1)
			{
				resolution_and_spacing.second.clear();
				logging_instance->QueueFileLogging("Image is static", m_log_file_id_, IO::Logging::NORMAL);
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

	std::vector<cv::Point> WSICS_Algorithm::GetTileCoordinates_(MultiResolutionImage& tiled_image, const std::vector<double>& spacing, const uint32_t tile_size, const uint32_t min_level)
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
		const std::vector<unsigned long long> dimensions = tiled_image.getLevelDimensions(number_of_levels - 1);
		uint32_t skip_factor = 1;
		float background_tissue_threshold = m_parameters_.background_threshold;
		uint32_t level_scale_difference = 1;

		std::vector<cv::Point> tile_coordinates;
		if (number_of_levels > 1)
		{
			logging_instance->QueueCommandLineLogging("Analyzing level: " + std::to_string(number_of_levels - 1), IO::Logging::NORMAL);

			const std::vector<unsigned long long> next_level_dimensions = tiled_image.getLevelDimensions(number_of_levels - 2);
			level_scale_difference = std::pow(std::round(next_level_dimensions[0] / next_level_dimensions[0]), 2);

			// Loops through each level, acquiring coordinates for each and reusing them to calculate the set of coordinates for a higher magnification.
			tile_coordinates = std::move(Misc::LevelReading::ReadLevelTiles(tiled_image, dimensions[0], dimensions[1], tile_size, number_of_levels - 1, skip_factor, background_tissue_threshold));
			for (char level_number = number_of_levels - 2; level_number >= 0; --level_number)
			{
				if (level_number != 0)
				{
					skip_factor = 2;

					const std::vector<unsigned long long> dimensions_level_low = tiled_image.getLevelDimensions(level_number);
					const std::vector<unsigned long long> dimensions_level_high = tiled_image.getLevelDimensions(level_number - 1);
					level_scale_difference = std::pow(std::floor(dimensions_level_high[0] / dimensions_level_low[0]), 2);
				}

				std::string log_text = "Analyzing level: " + std::to_string(level_number) + " - Tiles containing tissue: " + std::to_string(tile_coordinates.size());
				logging_instance->QueueCommandLineLogging(log_text, IO::Logging::NORMAL);
				logging_instance->QueueFileLogging(log_text, m_log_file_id_, IO::Logging::NORMAL);

				background_tissue_threshold -= 0.1;
				tile_coordinates = std::move(Misc::LevelReading::ReadLevelTiles(tiled_image, tile_coordinates, tile_size, level_number, skip_factor, level_scale_difference, background_tissue_threshold));
			}
		}
		else
		{
			tile_coordinates = std::move(Misc::LevelReading::ReadLevelTiles(tiled_image, dimensions[0], dimensions[1], tile_size, number_of_levels - 1, 0.9, skip_factor));
		}

		return tile_coordinates;
	}
}