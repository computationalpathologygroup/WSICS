#ifndef __WSICS_BLOBOPERATIONS__
#define __WSICS_BLOBOPERATIONS__
#include <unordered_map>

#include <opencv2/core/core.hpp>

#include "BLOB.h"

namespace WSICS::BLOB_Operations
{
	enum MaskType { FOUR_CONNECTEDNESS, EIGHT_CONNECTEDNESS };

	/// <summary>
	/// Detects and annotates BLOBS.
	/// </summary>
	/// <param name="binary_matrix">A matrix with binary values.</param>
	/// <param name="output_matrix">The matrix to write the results into.</param>
	/// <param name="mask_type">Whether to use four- or eight-connectedness.</param>
	/// <returns>The amount of labeled BLOBs within the matrix.</returns>
	size_t LabelBLOBs(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const MaskType mask_type);
	/// <summary>
	/// Detects and annotates BLOBS, adding additional label parameters into the referenced arrays.
	/// </summary>
	/// <param name="binary_matrix">A matrix with binary values.</param>
	/// <param name="output_matrix">The matrix to write the results into.</param>
	/// <param name="mask_type">Whether to use four- or eight-connectedness.</param>
	/// <param name="stats_array">The array to write additional label statistics into.</param>
	/// <param name="centroids_array">The array to write label centroids into.</param>
	/// <returns>The amount of labeled BLOBs within the matrix.</returns>
	size_t LabelBLOBs(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const MaskType mask_type, cv::Mat& stats_array, cv::Mat& centroids_array);
	/// <summary>
	/// Groups together pixels from the same BLOB.
	/// </summary>
	/// <param name="matrix">A matrix where each pixels value represents the label of its corresponding BLOB.</param>
	/// <returns>An unordered map containing all the labeled BLOBs.</returns>
	std::unordered_map<size_t, BLOB> GroupLabeledPixels(const cv::Mat& matrix);
	/// <summary>
	/// Groups together pixels from the same BLOB.
	/// </summary>
	/// <param name="matrix">A matrix where each pixels value represents the label of its corresponding BLOB.</param>
	/// <param name="stats_array">The array holding the labeled BLOB information.</param>
	/// <returns>An unordered map containing all the labeled BLOBs.</returns>
	std::unordered_map<size_t, BLOB> GroupLabeledPixels(const cv::Mat& matrix, const cv::Mat& stats_array);
	/// <summary>
	/// Detects and annotates BLOBs, adding them to an unordered map that joins their label with the pixel coordinates for each.
	/// This is an amalgation of the LabelBLOBs and GroupLabeledPixels methods.
	/// </summary>
	/// <param name="binary_matrix">A matrix with binary values.</param>
	/// <param name="output_matrix">The matrix to write the results into.</param>
	/// <param name="mask_type">Whether to use four- or eight-connectedness.</param>
	/// <returns>An unordered map containing all the labeled BLOBs.</returns>
	std::unordered_map<size_t, BLOB> LabelAndGroup(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const MaskType mask_type);
};
#endif // __WSICS_BLOBOPERATIONS__