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
	std::vector<cv::Mat> SplitBGR(const cv::Mat& source);
	void ApplyLUT(const cv::Mat& source, cv::Mat& destination, const cv::Mat& lut);
	void ApplyLUT(const unsigned char* source, unsigned char* destination, const cv::Mat& lut, const size_t tile_size);

	void WriteNormalizedWSI_(const boost::filesystem::path& input_file, const boost::filesystem::path& output_file, const cv::Mat& normalized_lut, const uint32_t tile_size);
	void WriteNormalizedWSI_(const cv::Mat& static_image, const boost::filesystem::path& output_file, const cv::Mat& normalized_lut);

	void WriteSampleNormalizedImagesForTesting_(const std::string output_filename, const cv::Mat& normalized_lut, const cv::Mat& tile_image, const uint32_t tile_size);
	void WriteSampleNormalizedImagesForTesting_(const boost::filesystem::path& output_directory, const cv::Mat& lut_image, MultiResolutionImage& tiled_image, const std::vector<cv::Point>& tile_coordinates, const uint32_t tile_size);
};
#endif // __NORMALIZEDOUTPUT_H__