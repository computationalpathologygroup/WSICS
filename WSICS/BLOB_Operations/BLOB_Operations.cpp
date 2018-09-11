#include "BLOB_Operations.h"

#include <opencv2/opencv.hpp>

#include "../Misc/MatrixOperations.h"

namespace WSICS::BLOB_Operations
{
	size_t LabelBLOBs(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const MaskType mask_type)
	{
		// Ensures the output matrix has the same capacity as the input matrix.
		if (output_matrix.cols != binary_matrix.cols || output_matrix.rows != binary_matrix.rows)
		{
			output_matrix.cols = binary_matrix.cols;
			output_matrix.rows = binary_matrix.rows;
		}

		// Selects the mask and acquires the BLOBs.
		unsigned char mask = 4;
		if (mask_type == MaskType::EIGHT_CONNECTEDNESS)
		{
			mask = 8;
		}

		return static_cast<size_t>(cv::connectedComponents(binary_matrix, output_matrix, mask));
	}

	size_t LabelBLOBs(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const MaskType mask_type, cv::Mat& stats_array, cv::Mat& centroids_array)
	{
		// Ensures the output matrix has the same capacity as the input matrix.
		Misc::MatrixOperations::PrepareOutputForInput(binary_matrix, output_matrix);

		// Selects the mask and acquires the BLOBs.
		unsigned char mask = 4;
		if (mask_type == MaskType::EIGHT_CONNECTEDNESS)
		{
			mask = 8;
		}

		return static_cast<size_t>(cv::connectedComponentsWithStats(binary_matrix, output_matrix, stats_array, centroids_array, mask));
	}

	std::unordered_map<size_t, BLOB> GroupLabeledPixels(const cv::Mat& matrix)
	{
		std::unordered_map<size_t, BLOB> labeled_blobs;

		// Acquires all non-zero pixels and loops through them, adding them to the correct label in the map.
		std::vector<cv::Point2f> non_zero_pixels;
		cv::findNonZero(matrix, non_zero_pixels);
		for (const cv::Point2f& point : non_zero_pixels)
		{
			size_t label = matrix.at<size_t>(point);
			auto iterator = labeled_blobs.find(label);

			if (iterator == labeled_blobs.end())
			{
				iterator = labeled_blobs.insert({ label, BLOB() }).first;
			}
			iterator->second.Add(point);
		}

		return labeled_blobs;
	}

	std::unordered_map<size_t, BLOB> GroupLabeledPixels(const cv::Mat& matrix, const cv::Mat& stats_array)
	{
		// Reserves enough space for the labels.
		std::unordered_map<size_t, BLOB> labeled_blobs;
		labeled_blobs.reserve((stats_array.rows));

		for (int label = 0; label < stats_array.rows; ++label)
		{
			labeled_blobs.insert({ label + 1, BLOB() });
		}

		// Acquires all non-zero pixels and loops through them, adding them to the correct label in the map.
		cv::Mat converted_matrix;
		matrix.convertTo(converted_matrix, CV_8UC1);

		std::vector<cv::Point> non_zero_pixels;
		cv::findNonZero(converted_matrix, non_zero_pixels);

		for (const cv::Point& point : non_zero_pixels)
		{
			auto iterator = labeled_blobs.find(matrix.at<int32_t>(point))->second.Add(point);
		}

		return labeled_blobs;
	}

	std::unordered_map<size_t, BLOB> LabelAndGroup(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const MaskType mask_type)
	{
		cv::Mat stat_array;
		cv::Mat centroid_array;
		LabelBLOBs(binary_matrix, output_matrix, mask_type, stat_array, centroid_array);

		return GroupLabeledPixels(output_matrix, stat_array);
	}
}