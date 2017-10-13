#ifndef __ASAP_IMAGEPROCESSING_BLOBOPERATIONS_BLOBWINDOW_H
#define __ASAP_IMAGEPROCESSING_BLOBOPERATIONS_BLOBWINDOW_H

#include <unordered_map>
#include <opencv2/core.hpp>

#include "BLOB_Operations.h"

namespace ASAP::Image_Processing::BLOB_Operations
{
	/// <summary>
	/// Provides windowed / tiled access to the BLOBs of a given matrix.
	/// Thread unsafe.
	/// </summary>
	class BLOB_Window
	{
		public:
			/// <summary>
			/// Default Constructor. Initializes an empty BLOB_Window. Stores the window size, which can be used for a proper initialization.
			/// </summary>
			/// <param name="window_size">The window size used to iterate over the matrix.</param>
			BLOB_Window(const uint32_t window_size);
			/// <summary>
			/// Initializes the BLOB_Window with a matrix that contains labeled pixels.
			/// </summary>
			/// <param name="window_size">The window size used to iterate over the matrix.</param>
			/// <param name="blob_matrix">A matrix that contains labeled pixels.</param>
			BLOB_Window(const uint32_t window_size, const cv::Mat& blob_matrix);
			/// <summary>
			/// Initializes the BLOB_Window with a matrix that contains labeled pixels and the statistics associated with each.
			/// </summary>
			/// <param name="window_size">The window size used to iterate over the matrix.</param>
			/// <param name="labeled_blob_matrix">A matrix that contains labeled pixels.</param>
			/// <param name="stats_array">A matrix containing the statistics for each labeled BLOB.</param>
			BLOB_Window(const uint32_t window_size, const cv::Mat& labeled_blob_matrix, cv::Mat& stats_array);
			/// <summary>
			/// Initializes the BLOB_Window with a binary matrix onto which a BLOB detection and labeling will be performed.
			/// </summary>
			/// <param name="window_size">The window size used to iterate over the matrix.</param>
			/// <param name="binary_matrix">A matrix with binary pixels.</param>
			/// <param name="output_matrix">The matrix that the BLOB labeling result should be written to.</param>
			/// <param name="mask_type">The type of mask to use for the BLOB detection.</param>
			BLOB_Window(const uint32_t window_size, const cv::Mat& binary_matrix, cv::Mat& output_matrix, const MaskType mask_type);

			/// <summary>
			/// Clears the information currently held by the BLOB_Window.
			/// </summary>
			void Clear(void);

			/// <summary>
			/// Returns all the acquired BLOBs that are present within the matrix.
			/// </summary>
			/// <returns>All the acquired BLOBs that are present within the matrix.</returns>
			const std::unordered_map<size_t, BLOB>& GetAllMatrixBLOBs(void) const;
			/// <summary>
			/// Acquires all the BLOBs that have any kind of overlap with the current window.
			/// </summary>
			/// <returns>All the BLOBs that have any kind of overlap with the current window.</returns>
			std::unordered_map<size_t, BLOB*> GetWindowBLOBs(void);

			/// <summary>
			/// Returns the current window or tile size.
			/// </summary>
			/// <returns>The current window size.</returns>
			uint32_t GetWindowSize(void) const;
			/// <summary>
			/// Replaces the current window size value and resets the window position to the beginning.
			/// </summary>
			/// <param name="window_size">The new window or tile size that is used to iterate over the matrix.</param>
			void SetWindowSize(const uint32_t window_size);

			/// <summary>
			/// Shifts the window a single window size towards the top left.
			/// Moves across the decreasing x axis until it hits the left edge, then across the decreasing y axis to place itself against the right edge again.
			/// </summary>
			/// <returns>Whether or not the window could be shifted backward.</returns>
			bool ShiftWindowBackward(void);
			/// <summary>
			///	Shifts the window a single window size towards the bottom right.
			/// Moves across the increasing x axis until it hits the right edge, then across the increasing y axis to place itself against the left edge again.
			/// </summary>
			/// <returns>Whether or not the window could be shifted forward.</returns>
			bool ShiftWindowForward(void);
			/// <summary>
			/// Shifts the window to its initial position at the top left corner of the matrix.
			/// </summary>
			void ShiftWindowToBegin(void);
			/// <summary>
			/// Shifts the window to its final position at the bottom right corner of the matrix.
			/// </summary>
			void ShiftWindowToEnd(void);

		private:
			uint32_t		m_window_size_;
			cv::Point2f m_window_top_left_;
			cv::Point2f m_window_bottom_right_;
			cv::Point2f m_matrix_bottom_right_;

			std::unordered_map<size_t, BLOB>		m_labeled_blobs_;
	};
};
#endif // __ASAP_IMAGEPROCESSING_BLOBOPERATIONS_BLOBWINDOW_H