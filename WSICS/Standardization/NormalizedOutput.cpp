#include "NormalizedOutput.h"

#include <opencv2/highgui.hpp>

#include "../IO/Logging/LogHandler.h"
#include "../Misc/LevelReading.h"
#include "../Misc/Random.h"
#include "../Misc/MT_Singleton.hpp"

namespace WSICS::Standardization
{
	std::vector<cv::Mat> SplitBGR(const cv::Mat& source)
	{
		std::vector<cv::Mat> bgr(3);
		cv::split(source, bgr);
		for (cv::Mat& channel : bgr)
		{
			channel.convertTo(channel, CV_8UC1);
		}
		return bgr;
	}

	void ApplyLUT(const cv::Mat& source, cv::Mat& destination, const cv::Mat& lut)
	{
		std::vector<cv::Mat> source_channels(SplitBGR(source));
		std::vector<cv::Mat> lut_channels(SplitBGR(lut));

		for (size_t row = 0; row < source.rows; ++row)
		{
			for (size_t col = 0; col < source.cols; ++col)
			{
				size_t index = 256 * 256 * source_channels[0].at<uchar>(row, col) +
							   256 * source_channels[1].at<uchar>(row, col) +
							   source_channels[2].at<uchar>(row, col);

				source_channels[0].at<uchar>(row, col) = lut_channels[2].at<uchar>(index, 0);
				source_channels[1].at<uchar>(row, col) = lut_channels[1].at<uchar>(index, 0);
				source_channels[2].at<uchar>(row, col) = lut_channels[0].at<uchar>(index, 0);
			}
		}

		cv::merge(source_channels, destination);
	}

	void ApplyLUT(const unsigned char* source, unsigned char* destination, const cv::Mat& lut, const size_t tile_size)
	{
		std::vector<cv::Mat> lut_channels(SplitBGR(lut));

		size_t source_index			= 0;
		size_t destination_index	= 0;
		for (size_t pixel = 0; pixel < tile_size * tile_size; ++pixel)
		{
			size_t index = 256 * 256 * source[source_index] + 256 * source[source_index + 1] + source[source_index + 2];
			destination[destination_index++] = lut_channels[2].at<unsigned char>(index, 0);
			destination[destination_index++] = lut_channels[1].at<unsigned char>(index, 0);
			destination[destination_index++] = lut_channels[0].at<unsigned char>(index, 0);
			source_index += 3;
		}
	}

	void WriteNormalizedWSI(const boost::filesystem::path& input_file, const boost::filesystem::path& output_file, const cv::Mat& normalized_lut, const uint32_t tile_size)
	{
		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

		MultiResolutionImageReader reader;
		MultiResolutionImage* tiled_image = reader.open(input_file.string());
		const std::vector<unsigned long long> dimensions = tiled_image->getLevelDimensions(0);

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

			ApplyLUT(data, data, normalized_lut, tile_size);
			image_writer.writeBaseImagePart((void*)data);

			delete[] data;
		}

		logging_instance->QueueCommandLineLogging("Finalizing images", IO::Logging::NORMAL);
		image_writer.finishImage();
	}

	void WriteNormalizedWSI(const cv::Mat& static_image, const boost::filesystem::path& output_file, const cv::Mat& normalized_lut)
	{
		cv::Mat normalized_image;
		ApplyLUT(static_image, normalized_image, normalized_lut);
		cv::imwrite(output_file.string(), normalized_image);

		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
		logging_instance->QueueCommandLineLogging("Normalized image written to: " + output_file.string(), IO::Logging::NORMAL);
	}

	void WriteNormalizedSample(const std::string output_filepath, const cv::Mat& normalized_lut, const cv::Mat& tile_image, const uint32_t tile_size)
	{
		cv::Mat lut_slide_image;
		ApplyLUT(tile_image, lut_slide_image, normalized_lut);
		cv::imwrite(output_filepath, lut_slide_image);

		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
		logging_instance->QueueCommandLineLogging("Normalized image is written...", IO::Logging::NORMAL);
	}

	void WriteNormalizedSamples(
		const boost::filesystem::path& output_directory,
		const cv::Mat& normalized_lut,
		MultiResolutionImage& tiled_image,
		const std::vector<cv::Point>& tile_coordinates,
		const uint32_t tile_size)
	{
		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

		logging_instance->QueueCommandLineLogging("Writing sample standardized images in: " + output_directory.string(), IO::Logging::NORMAL);

		std::vector<size_t> random_integers(Misc::Random::CreateListOfRandomIntegers(tile_coordinates.size(), Misc::MT_Singleton::GetGenerator()));

		size_t num_to_write = 20 > tile_coordinates.size() ? tile_coordinates.size() : 20;
		cv::Mat tile_image(cv::Mat::zeros(tile_size, tile_size, CV_8UC3));
		for (size_t tile = 0; tile < num_to_write; ++tile)
		{
			unsigned char* data = nullptr;
			tiled_image.getRawRegion(tile_coordinates[random_integers[tile]].x * tiled_image.getLevelDownsample(0), tile_coordinates[random_integers[tile]].y * tiled_image.getLevelDownsample(0), tile_size, tile_size, 0, data);
			cv::Mat tile_image = cv::Mat::zeros(tile_size, tile_size, CV_8UC3);
			Misc::LevelReading::ArrayToMatrix(data, tile_image, 0);
			delete[] data;

			ApplyLUT(tile_image, tile_image, normalized_lut);
			std::string filename_lut(output_directory.string() + "/" + "tile_" + std::to_string(random_integers[tile]) + "_normalized.tif");
			logging_instance->QueueCommandLineLogging(filename_lut, IO::Logging::NORMAL);
			cv::imwrite(filename_lut, tile_image);
		}

		logging_instance->QueueCommandLineLogging("Sample images are written...", IO::Logging::NORMAL);
	}
}