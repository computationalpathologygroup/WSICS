#ifndef __HSD_BackgroundMask_H__
#define __HSD_BackgroundMask_H__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "HSD_Model.h"

/// <summary>Contains functions that can be applied to create or transform background masks.</summary>
namespace WSICS::HSD::BackgroundMask
{
	/// <summary>Creates a background mask based on several channels and a user defined threshold.</summary>
	/// <param name="hsd_image">The HSD Model image, from which to create the background mask.</param>
	/// <param name="global_threshold">The threshold to apply to the density matrix.</param>
	/// <param name="channel_threshold">The threshold to apply to the channel matrices.</param>
	/// <returns>A cv::Mat containing the created background mask.</returns>
	cv::Mat CreateBackgroundMask(const HSD_Model& hsd_image, const float global_threshold, const float channel_threshold);

	/// <summary>Counts the amount of background pixels in the mask.</summary>
	/// <returns>The amount of background pixels.</returns>
	size_t CountBackGroundPixels(cv::Mat& background_mask);

	/// <summary>Counts the pixels not part of the background mask.</summary>
	/// <returns>The amount of non-background pixels.</returns>
	size_t CountNonBackGroundPixels(cv::Mat& background_mask);
}
#endif // __HSD_BackgroundMask_H__