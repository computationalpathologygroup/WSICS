#pragma once
#include <opencv2/core/core.hpp>

namespace ASAP::MiscMatrixOperations
{
	/// <summary>
	/// Ensures the output matrix equals the type and size of the input matrix.
	/// </summary>
	/// <param name="input_matrix">The input matrix to acquire the parameters from.</param>
	/// <param name="output_matrix">The output matrix to prepare.</param>
	void PrepareOutputForInput(const cv::Mat& input_matrix, cv::Mat& output_matrix);
}