#include "BLOB_Operations.h"

#include <opencv2/opencv.hpp>

namespace ASAP::Image_Processing::BLOB_Operations
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

		return static_cast<size_t>(cv::connectedComponents(binary_matrix, output_matrix, mask, CV_16U));
	}

	size_t LabelBLOBs(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const MaskType mask_type, cv::Mat& stats_array, cv::Mat& centroids_array)
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

		return static_cast<size_t>(cv::connectedComponentsWithStats(binary_matrix, output_matrix, stats_array, centroids_array, mask, CV_16U));
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

	std::unordered_map<size_t, BLOB> GroupLabeledPixels(const cv::Mat& matrix, cv::Mat& stats_array)
	{
		// Reserves enough space for the labels.
		std::unordered_map<size_t, BLOB> labeled_blobs;
		labeled_blobs.reserve((stats_array.rows * stats_array.cols) - 1);

		for (int label = 1; label < stats_array.rows; ++label)
		{
			uint32_t* data_ptr(stats_array.ptr<uint32_t>(label));

			cv::Point2f top_left(data_ptr[cv::ConnectedComponentsTypes::CC_STAT_LEFT], data_ptr[cv::ConnectedComponentsTypes::CC_STAT_TOP]);
			cv::Point2f bottom_right(top_left.x + data_ptr[cv::ConnectedComponentsTypes::CC_STAT_WIDTH], top_left.y + data_ptr[cv::ConnectedComponentsTypes::CC_STAT_HEIGHT]);

			labeled_blobs.insert({ label, BLOB(top_left, bottom_right) });
		}

		// Acquires all non-zero pixels and loops through them, adding them to the correct label in the map.
		std::vector<cv::Point> non_zero_pixels;
		cv::findNonZero(matrix, non_zero_pixels);
		for (const cv::Point& point : non_zero_pixels)
		{
			auto iterator = labeled_blobs.find(matrix.at<size_t>(point))->second.Add(point);
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