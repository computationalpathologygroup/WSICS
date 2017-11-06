#include "LevelReading.h"

#include <memory>

namespace LevelReading
{
	size_t ArrayToMatrix(unsigned char* data, cv::Mat& matrix, bool consider_background, bool is_tiff)
	{
		size_t background_count = 0;
		size_t base_index = 0;
		for (int row = 0; row < matrix.rows; ++row)
		{
			uchar* row_ptr = matrix.ptr(row);
			for (int col = 0; col < matrix.cols; ++col)
			{
				// Points to each pixel value in turn assuming a CV_8UC1 greyscale image.
				if (is_tiff)
				{
					*row_ptr = data[base_index + 2];
					*(row_ptr + 1) = data[base_index + 1];
					*(row_ptr + 2) = data[base_index];
				}
				else
				{
					*row_ptr = data[base_index];
					*(row_ptr + 1) = data[base_index + 1];
					*(row_ptr + 2) = data[base_index + 2];
				}

				row_ptr += 3;

				// Threshold was 230 for mrxs files
				if (consider_background && ((data[base_index] > 200 && data[base_index + 1] > 200 && data[base_index + 2] > 200) || (data[base_index] == 0 && data[base_index + 1] == 0 && data[base_index + 2] == 0)))
				{
					++background_count;
				}

				base_index += is_tiff ? 3 : 4;
			}
		}

		return background_count;
	}

	std::vector<cv::Point> GetNextLevelCoordinates(std::vector<cv::Point>& current_level_coordinates, uint32_t tile_size, int32_t scale_diff)
	{
		std::vector<cv::Point> next_level_tile_coordinates;
		next_level_tile_coordinates.reserve(current_level_coordinates.size() * scale_diff * scale_diff);

		scale_diff = std::sqrt(scale_diff);
		for (cv::Point& point : current_level_coordinates)
		{
			for (int r = 0; r < scale_diff; ++r)
			{
				for (int c = 0; c < scale_diff; ++c)
				{
					next_level_tile_coordinates.push_back({ static_cast<int>(scale_diff * point.x + r * tile_size), static_cast<int>(scale_diff * point.y + c * tile_size) });
				}
			}
		}

		return next_level_tile_coordinates;
	}

	std::vector<cv::Point> ReadLevelTiles(MultiResolutionImage& tiled_image, size_t x_dimension, size_t y_dimension, uint32_t tile_size, bool is_tiff, uint32_t level, float background_threshold, uint32_t skip_factor)
	{
		unsigned char* data(new unsigned char[tile_size * tile_size * 4]);

		cv::Mat tile_image(cv::Mat::zeros(tile_size, tile_size, CV_8UC3));

		std::vector<cv::Point> tile_coordinates;
		for (int y = 0; y < y_dimension; y += (tile_size)* skip_factor)
		{
			for (int x = 0; x < x_dimension; x += tile_size)
			{
				tiled_image.getRawRegion(x  *tiled_image.getLevelDownsample(level), y * tiled_image.getLevelDownsample(level), tile_size, tile_size, level, data);

				size_t background_count = ArrayToMatrix(data, tile_image, true, is_tiff);
				if ((float)background_count / (tile_size * tile_size) < background_threshold)
				{
					tile_coordinates.push_back({ x, y });
				}
			}
		}

		delete[] data;

		return tile_coordinates;
	}

	std::vector<cv::Point> ReadLevelTiles(
		MultiResolutionImage& tiled_image,
		std::vector<cv::Point> current_tile_coordinates,
		uint32_t tile_size,
		bool is_tiff,
		uint32_t level,
		float background_threshold,
		uint32_t skip_factor,
		int32_t scale_diff)
	{
		unsigned char* data(new unsigned char[tile_size * tile_size * 4]);

		cv::Mat tile_image(cv::Mat::zeros(tile_size, tile_size, CV_8UC3));

		std::vector<cv::Point> next_level_tile_coordinates(GetNextLevelCoordinates(current_tile_coordinates, tile_size, scale_diff));

		std::vector<cv::Point> tile_coordinates;
		for (int i = 0; i < next_level_tile_coordinates.size(); i += skip_factor)
		{
			tiled_image.getRawRegion(next_level_tile_coordinates[i].x * tiled_image.getLevelDownsample(level), next_level_tile_coordinates[i].y * tiled_image.getLevelDownsample(level), tile_size, tile_size, level, data);

			size_t background_count = ArrayToMatrix(data, tile_image, true, is_tiff);
			if ((float)background_count / (tile_size * tile_size) < background_threshold)
			{
				tile_coordinates.push_back(next_level_tile_coordinates[i]);
			}
		}

		delete[] data;

		return tile_coordinates;
	}
}