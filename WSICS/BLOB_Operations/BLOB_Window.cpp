#include "BLOB_Window.h"

#include <cmath>
#include <stdexcept>

namespace WSICS::BLOB_Operations
{
	BLOB_Window::BLOB_Window(const uint32_t window_size) :
		m_window_size_(window_size), m_window_step_size_(CalculateWindowStepSize_(window_size))
	{
	}

	BLOB_Window::BLOB_Window(const uint32_t window_size, const cv::Mat& blob_matrix) :
		m_window_size_(window_size),
		m_window_step_size_(CalculateWindowStepSize_(window_size)),
		m_window_top_left_(0, 0),
		m_window_bottom_right_(m_window_size_, m_window_size_),
		m_matrix_bottom_right_(blob_matrix.rows - 1, blob_matrix.cols - 1),
		m_labeled_blobs_(BLOB_Operations::GroupLabeledPixels(blob_matrix))
	{
	}

	BLOB_Window::BLOB_Window(const uint32_t window_size, const cv::Mat& labeled_blob_matrix, const cv::Mat& stats_array) :
		m_window_size_(window_size),
		m_window_step_size_(CalculateWindowStepSize_(window_size)),
		m_window_top_left_(0, 0),
		m_window_bottom_right_(m_window_size_, m_window_size_),
		m_matrix_bottom_right_(labeled_blob_matrix.rows - 1, labeled_blob_matrix.cols - 1),
		m_labeled_blobs_(BLOB_Operations::GroupLabeledPixels(labeled_blob_matrix, stats_array))
	{
	}

	BLOB_Window::BLOB_Window(const uint32_t window_size, const cv::Mat& binary_matrix, cv::Mat& output_matrix, const MaskType mask_type) : 
		m_window_size_(window_size),
		m_window_step_size_(CalculateWindowStepSize_(window_size)),
		m_window_top_left_(0, 0), 
		m_window_bottom_right_(m_window_size_, m_window_size_), 
		m_matrix_bottom_right_(binary_matrix.rows - 1, binary_matrix.cols - 1),
		m_labeled_blobs_(BLOB_Operations::LabelAndGroup(binary_matrix, output_matrix, mask_type))
	{
	}

	void BLOB_Window::Clear(void)
	{
		m_window_top_left_		= cv::Point2f();
		m_window_bottom_right_	= cv::Point2f();
		m_matrix_bottom_right_	= cv::Point2f();
		m_labeled_blobs_.clear();
	}

	const std::unordered_map<size_t, BLOB>& BLOB_Window::GetAllMatrixBLOBs(void) const
	{
		return m_labeled_blobs_;
	}

	std::unordered_map<size_t, BLOB*> BLOB_Window::GetWindowBLOBs(void)
	{
		if (m_labeled_blobs_.empty())
		{
			throw std::out_of_range("This BLOB Window hasn't been initialized with any available BLOBs.");
		}

		std::unordered_map<size_t, BLOB*> blobs_within_window;
		for (std::pair<const size_t, BLOB>& labeled_blob : m_labeled_blobs_)
		{
			if (labeled_blob.second.BoxIntersectsWith(m_window_top_left_, m_window_bottom_right_))
			{
				blobs_within_window.insert({ labeled_blob.first, &labeled_blob.second });
			}
		}
		return blobs_within_window;
	}

	uint32_t BLOB_Window::GetWindowSize(void) const
	{
		return m_window_size_;
	}

	void BLOB_Window::SetWindowSize(const uint32_t window_size)
	{
		m_window_size_		= window_size;
		m_window_step_size_ = CalculateWindowStepSize_(window_size);
		ShiftWindowToBegin();
	}

	bool BLOB_Window::ShiftWindowBackward(void)
	{
		// If the window reached the left edge.
		if (m_window_top_left_.x == 0)
		{
			// If the window already reached the top left corner.
			if (m_window_top_left_.y == 0)
			{
				return false;
			}

			m_window_top_left_.x = std::floor(m_matrix_bottom_right_.x / m_window_step_size_) * m_window_step_size_;
			m_window_top_left_.y = m_window_top_left_.y - m_window_step_size_;
			
		}
		else
		{
			m_window_top_left_.x = m_window_top_left_.x - m_window_step_size_;
			m_window_top_left_.y = m_window_top_left_.y;
		}

		m_window_bottom_right_.x		= m_window_top_left_.x + m_window_size_;
		m_window_bottom_right_.y		= m_window_top_left_.y + m_window_size_;

		return true;
	}

	bool BLOB_Window::ShiftWindowForward(void)
	{
		// Shifts the window towards the right.
		cv::Point2f new_top_left(m_window_top_left_.x + m_window_step_size_, m_window_top_left_.y);

		// If the new point is beyond the x axis range of the matrix.
		if (new_top_left.x > m_matrix_bottom_right_.x)
		{
			// Shifts the window towards the bottom.
			new_top_left.x = 0;
			new_top_left.y = m_window_top_left_.y + m_window_step_size_;

			// If the range of both the x and y axis from the matrix have been exceeded.
			if (new_top_left.y > m_matrix_bottom_right_.y)
			{
				return false;
			}
		}

		// Adjust the current window to one defined by new_top_left + m_window_size_.
		m_window_top_left_ = new_top_left;
		m_window_bottom_right_.x = m_window_top_left_.x + m_window_size_;
		m_window_bottom_right_.y = m_window_top_left_.y + m_window_size_;

		return true;
	}

	void BLOB_Window::ShiftWindowToBegin(void)
	{
		m_window_top_left_		= cv::Point2f(0, 0);
		m_window_bottom_right_	= cv::Point2f(m_window_size_, m_window_size_);
	}

	void BLOB_Window::ShiftWindowToEnd(void)
	{
		m_window_top_left_		= cv::Point2f(std::floor(m_matrix_bottom_right_.x / m_window_step_size_) * m_window_step_size_,
											std::floor(m_matrix_bottom_right_.y / m_window_step_size_) * m_window_step_size_);
		m_window_bottom_right_	= cv::Point2f(m_window_top_left_.x + m_window_size_, m_window_top_left_.y + m_window_size_);
	}

	uint32_t BLOB_Window::CalculateWindowStepSize_(const uint32_t window_size)
	{
		return window_size * 5.0f / 6.0f;
	}
}