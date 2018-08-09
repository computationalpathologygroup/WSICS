#include "BackgroundMask.h"

namespace HSD::BackgroundMask
{
	cv::Mat CreateBackgroundMask(const HSD_Model& hsd_image, const float global_threshold, const float channel_threshold)
	{
		// Creates empty masks.
		cv::Mat density_mask		= cv::Mat::zeros(hsd_image.density.size(), CV_8UC1);
		cv::Mat red_density_mask	= density_mask;
		cv::Mat green_density_mask	= density_mask;
		cv::Mat blue_density_mask	= density_mask;
		cv::Mat dark_density_mask	= density_mask;

		// Calculates the masks for the density and each channel.
		cv::threshold(hsd_image.density, density_mask, global_threshold, 1, 1);
		cv::threshold(hsd_image.red_density, red_density_mask, channel_threshold, 1, 1);
		cv::threshold(hsd_image.green_density, green_density_mask, channel_threshold, 1, 1);
		cv::threshold(hsd_image.blue_density, blue_density_mask, channel_threshold, 1, 1);

		// Considers pure black pixels as background.
		cv::threshold(hsd_image.density, dark_density_mask, 5.5, 1, 1);
		dark_density_mask = 1 - dark_density_mask;

		// Creates the background mask object and returns it.
		cv::Mat background_mask(std::move(blue_density_mask.mul(green_density_mask.mul(red_density_mask.mul(density_mask))) + dark_density_mask));
		background_mask.convertTo(background_mask, CV_8UC1);

		return background_mask;
	}

	// Counts the amount of background pixels.
	size_t CountBackGroundPixels(cv::Mat& background_mask)
	{
		size_t background_pixels = 0;
		for (size_t row = 0; row < background_mask.rows; ++row)
		{
			for (size_t column = 0; column < background_mask.cols; ++column)
			{
				background_pixels += static_cast<unsigned char>(background_mask.at<unsigned char>(row, column) != 1);
			}
		}
		return background_pixels;
	}

	// Counts the amount of non-background pixels.
	size_t CountNonBackGroundPixels(cv::Mat& background_mask)
	{
		return (background_mask.rows * background_mask.cols) - CountBackGroundPixels(background_mask);
	}
}