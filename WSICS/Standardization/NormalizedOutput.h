#ifndef __NORMALIZEDOUTPUT_H__
#define __NORMALIZEDOUTPUT_H__

#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>

#include "multiresolutionimageinterface/MultiResolutionImage.h"
#include "multiresolutionimageinterface/MultiResolutionImageReader.h"
#include "multiresolutionimageinterface/MultiResolutionImageWriter.h"
#include "multiresolutionimageinterface/OpenSlideImageFactory.h"

namespace StainNormalization
{
	/// <summary>
	/// Splits a BGR matrix and ensures each split matrix has the correct type for the normalization of an image.
	/// </summary>
	/// <param name="source">The matrix to split.</param>
	/// <returns>A vector containing the seperated matrices.</returns>
	std::vector<cv::Mat> SplitBGR(const cv::Mat& source);
	/// <summary>
	/// Applies a LUT onto a source matrix.
	/// </summary>
	/// <param name="source">The matrix to apply the LUT to.</param>
	/// <param name="destination">The result matrix.</param>
	/// <param name="lut">The LUT to apply.</param>
	void ApplyLUT(const cv::Mat& source, cv::Mat& destination, const cv::Mat& lut);
	/// <summary>
	/// Applies the LUT to a unsigned char array.
	/// </summary>
	/// <param name="source">The source array.</param>
	/// <param name="destination">The array to write the result to.</param>
	/// <param name="lut">The LUT to apply.</param>
	/// <param name="tile_size">The tile size of the original WSI.</param>
	void ApplyLUT(const unsigned char* source, unsigned char* destination, const cv::Mat& lut, const size_t tile_size);

	/// <summary>
	/// Writes a normalized WSI to the passed file path.
	/// </summary>
	/// <param name="input_file">The original WSI file path.</param>
	/// <param name="output_file">The file path for the resulting output WSI.</param>
	/// <param name="normalized_lut">The LUT to use for the normalization of the WSI.</param>
	/// <param name="tile_size">The tile size of the original WSI.</param>
	void WriteNormalizedWSI(const boost::filesystem::path& input_file, const boost::filesystem::path& output_file, const cv::Mat& normalized_lut, const uint32_t tile_size);
	/// <summary>
	/// Writes a normalized WSI to the passed file path.
	/// </summary>
	/// <param name="static_image">The static patch matrix.</param>
	/// <param name="output_file">The file path for the resulting output WSI.</param>
	/// <param name="normalized_lut">The tile size of the original WSI.</param>
	void WriteNormalizedWSI(const cv::Mat& static_image, const boost::filesystem::path& output_file, const cv::Mat& normalized_lut);

	/// <summary>
	/// Writes small sample of the normalized WSI.
	/// </summary>
	/// <param name="output_filename">The file path to where the sample should be written.</param>
	/// <param name="normalized_lut">The LUT to normalize the sample with.</param>
	/// <param name="tile_image">The original image to select the tile from.</param>
	/// <param name="tile_size">The tile size of the original WSI.</param>
	void WriteNormalizedSample(const std::string output_filename, const cv::Mat& normalized_lut, const cv::Mat& tile_image, const uint32_t tile_size);
	/// <summary>
	/// Writes small samples of the normalized WSI.
	/// </summary>
	/// <param name="output_directory">The directory path to which the samples should be written.</param>
	/// <param name="lut_image">The LUT to normalize the samples with.</param>
	/// <param name="tiled_image">The image to select the samples from.</param>
	/// <param name="tile_coordinates">The coordinates for each tile within the image.</param>
	/// <param name="tile_size">The size of each tile.</param>
	void WriteNormalizedSamples(const boost::filesystem::path& output_directory, const cv::Mat& lut_image, MultiResolutionImage& tiled_image, const std::vector<cv::Point>& tile_coordinates, const uint32_t tile_size);
};
#endif // __NORMALIZEDOUTPUT_H__