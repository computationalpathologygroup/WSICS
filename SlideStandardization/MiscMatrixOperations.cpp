#include "MiscMatrixOperations.h"

namespace ASAP::MiscMatrixOperations
{
	void PrepareOutputForInput(const cv::Mat& input_matrix, cv::Mat& output_matrix)
	{
		if (input_matrix.data != output_matrix.data)
		{
			if (input_matrix.size() != output_matrix.size())
			{
				output_matrix = cv::Mat::zeros(input_matrix.rows, input_matrix.cols, input_matrix.type());
			}
			else if (input_matrix.type() != output_matrix.type())
			{
				output_matrix.convertTo(output_matrix, input_matrix.type());
			}
		}
	}
}