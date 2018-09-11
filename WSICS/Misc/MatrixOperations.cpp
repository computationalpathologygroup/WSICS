#include "MatrixOperations.h"

namespace WSICS::Misc::MatrixOperations
{
	void NormalizeMat(const cv::Mat& source, cv::Mat& destination)
	{
		if (source.data != destination.data)
		{
			source.copyTo(destination);
		}

		double min, max, positive_min;

		cv::minMaxLoc(destination, &min, &max);
		positive_min = std::sqrt(std::pow(min, 2));

		destination += positive_min;
		destination.convertTo(destination, source.type(), 1.0f / (max + positive_min));
	}

	std::vector<cv::Mat> NormalizeMats(const std::vector<cv::Mat>& sources)
	{
		double lowest	= 0;
		double highest	= 0;

		std::vector<cv::Mat> normalized(sources.size());
		for (size_t source = 0; source < sources.size(); ++source)
		{
			double min, max;
			cv::minMaxLoc(sources[source], &min, &max);
			if (min < lowest)
			{
				lowest = min;
			}
			if (max > highest)
			{
				highest = max;
			}

			sources[source].copyTo(normalized[source]);
		}

		lowest = std::sqrt(std::pow(lowest, 2)); // Definess the minimum of the range.
		highest = std::sqrt(std::pow(highest, 2)) + lowest; // Defines the maximum of the range.
		
		for (cv::Mat& matrix : normalized)
		{
			matrix += lowest;
			matrix.convertTo(matrix, matrix.type(), 1.0f / highest);
		}

		return normalized;
	}

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