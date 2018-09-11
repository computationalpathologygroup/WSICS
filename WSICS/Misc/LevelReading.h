#ifndef __LevelReading_H__
#define __LevelReading_H__

#include <opencv2/core/core.hpp>

#include "multiresolutionimageinterface/MultiResolutionImage.h"

/// <summary>
///
/// </summary>
namespace WSICS::Misc::LevelReading
{
	/// <summary>
	/// Inserts the data from a vector or array into an OpenCV matrix. Assumes the vector or array are at least compareable to the matrix in terms of size.
	/// </summary>
	/// <param name="data">The starting element of a vector or array.</param>
	/// <param name="output">The matrix to transfer the data to.</param>
	/// <param name="consider_background">Whether or not to make an attempt at background pixel counting.</param>
	/// <returns>The counted background pixels, if applicable. Otherwise, 0.</returns>
	size_t ArrayToMatrix(unsigned char* data, cv::Mat& output, const bool consider_background);
	/// <summary>
	/// Acquires the tile coordinates for the next level within the pyramid.
	/// </summary>
	/// <param name="current_level_coordinates">The level coordinates for the current tile.</param>
	/// <param name="tile_size">The size of the tiles.</param>
	/// <param name="scale_diff">The scale difference between each level.</param>
	/// <returns>A vector containing the coordinates for the next level, based on the passsed coordinates.</returns>
	std::vector<cv::Point> GetNextLevelCoordinates(std::vector<cv::Point>& current_level_coordinates, uint32_t tile_size, int32_t scale_diff);
	/// <summary>
	/// Acquires the tile coordinates for the level passed.
	/// </summary>
	/// <param name="tiled_image">The tiled image to extract the coordinates from.</param>
	/// <param name="x_dimension">The size of the x dimension to search within.</param>
	/// <param name="y_dimension">The size of the y dimension to search within,</param>
	/// <param name="tile_size">The size of each tile.</param>
	/// <param name="level">The level to select the coordinates for.</param>
	/// <param name="skip_factor">Is added to each iterator of the coordinate search. Enabling reduction in the coherence of coordinate selection.</param>
	/// <param name="background_threshold">The pixel value to consider background, and thus not include.</param>
	/// <returns>A vector containing the selected tile coordinates.</returns>
	std::vector<cv::Point> ReadLevelTiles(
		MultiResolutionImage& tiled_image,
		std::vector<cv::Point> current_tile_coordinates,
		const uint32_t tile_size,
		const uint32_t level,
		const uint32_t skip_factor,
		const int32_t scale_diff,
		const float background_threshold);
	/// <summary>
	/// Acquires the tile coordinates based on the immediate next level.
	/// </summary>
	/// <param name="tiled_image">The tiled image to extract pixel information from.</param>
	/// <param name="current_tile_coordinates">The tile coordinates for the current level.</param>
	/// <param name="tile_size">The size of each tile.</param>
	/// <param name="level">The level to select the coordinates for.</param>
	/// <param name="skip_factor">Is added to each iterator of the coordinate search. Enabling reduction in the coherence of coordinate selection.</param>
	/// <param name="scale_diff">The scale difference each level.</param>
	/// <param name="background_threshold">he pixel value to consider background, and thus not include.</param>
	/// <returns>A vector containing the selected tile coordinates.</returns>
	std::vector<cv::Point> ReadLevelTiles(
		MultiResolutionImage& tiled_image,
		const size_t x_dimension,
		const size_t y_dimension,
		const uint32_t tile_size,
		const uint32_t level,
		const uint32_t skip_factor,
		const float background_threshold);
};
#endif //__LevelReading_H__