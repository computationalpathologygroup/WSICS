#pragma once
#include <opencv2/core/core.hpp>

namespace ASAP::MiscMatrixOperations
{
	/// <summary>
	/// Normalizes the Mat to ensure the values range from 0 to 1.
	/// Warning: Acquires the matrix type from the source matrix.
	/// </summary>
	/// <param name="source">The input matrix to normalize.</param>
	/// <param name="destination">The output matrix to write the results to.</param>
	void NormalizeMat(const cv::Mat& source, cv::Mat& destination);

	std::vector<cv::Mat> NormalizeMats(const std::vector<cv::Mat>& sources);

	/// <summary>
	/// Ensures the output matrix equals the type and size of the input matrix.
	/// </summary>
	/// <param name="source">The input matrix to acquire the parameters from.</param>
	/// <param name="destination">The output matrix to prepare.</param>
	void PrepareOutputForInput(const cv::Mat& source, cv::Mat& destination);
}