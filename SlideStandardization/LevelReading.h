#ifndef __LevelReading_H__
#define __LevelReading_H__

#include <opencv2/core/core.hpp>

#include "multiresolutionimageinterface/MultiResolutionImage.h"

/// <summary>
///
/// </summary>
namespace LevelReading
{
	/// <summary>
	/// Inserts the data from a vector or array into an OpenCV matrix. Assumes the vector or array are at least compareable to the matrix in terms of size.
	/// </summary>
	/// <param name="data">The starting element of a vector or array.</param>
	/// <param name="matrix">The matrix to transfer the data to.</param>
	/// <param name="consider_background">Whether or not to make an attempt at background pixel counting.</param>
	/// <param name="is_tiff">Whether or not the image is of the TIFF format.</param>
	/// <returns>The counted background pixels, if applicable. Otherwise, 0.</returns>
	size_t ArrayToMatrix(unsigned char* data, cv::Mat& matrix, bool consider_background, bool is_tiff);
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
	/// <param name="is_tiff">Whether or not the image is of the TIFF format.</param>
	/// <param name="level">The level to select the coordinates for.</param>
	/// <param name="background_threshold">The pixel value to consider background, and thus not include.</param>
	/// <param name="skip_factor">Is added to each iterator of the coordinate search. Enabling reduction in the coherence of coordinate selection.</param>
	/// <returns>A vector containing the selected tile coordinates.</returns>
	std::vector<cv::Point> ReadLevelTiles(MultiResolutionImage& tiled_image, size_t x_dimension, size_t y_dimension, uint32_t tile_size, bool is_tiff, uint32_t level, float background_threshold, uint32_t skip_factor);
	/// <summary>
	/// Acquires the tile coordinates based on the immediate next level.
	/// </summary>
	/// <param name="tiled_image">The tiled image to extract pixel information from.</param>
	/// <param name="current_tile_coordinates">The tile coordinates for the current level.</param>
	/// <param name="tile_size">The size of each tile.</param>
	/// <param name="is_tiff">Whether or not the image if of the TIFF format.</param>
	/// <param name="level">The level to select the coordinates for.</param>
	/// <param name="background_threshold">he pixel value to consider background, and thus not include.</param>
	/// <param name="skip_factor">Is added to each iterator of the coordinate search. Enabling reduction in the coherence of coordinate selection.</param>
	/// <param name="scale_diff">The scale difference each level.</param>
	/// <returns>A vector containing the selected tile coordinates.</returns>
	std::vector<cv::Point> ReadLevelTiles(MultiResolutionImage& tiled_image, std::vector<cv::Point> current_tile_coordinates, uint32_t tile_size, bool is_tiff, uint32_t level, float background_threshold, uint32_t skip_factor, int32_t scale_diff);
};
#endif //__LevelReading_H__