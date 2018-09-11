#include "BLOB.h"

#include <cmath>

namespace WSICS::BLOB_Operations
{
	BLOB::BLOB(void)
	{
	}

	BLOB::BLOB(const cv::Point2f& top_left, const cv::Point2f& bottom_right) : m_top_left_(top_left), m_bottom_right_(bottom_right)
	{
	}

	BLOB::BLOB(const std::vector<cv::Point2f>& blob_points) : m_points_(blob_points)
	{
	}

	BLOB::BLOB(const std::vector<cv::Point2f>& blob_points, const cv::Point2f& top_left, const cv::Point2f& bottom_right) 
		: m_points_(blob_points), m_top_left_(top_left), m_bottom_right_(bottom_right)
	{
	}

	BLOB::BLOB(std::vector<cv::Point2f>&& blob_points, const cv::Point2f& top_left, const cv::Point2f& bottom_right)
		: m_points_(std::move(blob_points)), m_top_left_(top_left), m_bottom_right_(bottom_right)
	{
	}

    bool BLOB::Add(const cv::Point2f& point)
    {
		// Checks whether or not this point already exists within the vector.
		size_t occurances = 0;
		for (const cv::Point2f& inserted_point : m_points_)
		{
			occurances += point == inserted_point;
		}
		if (occurances > 0)
		{
			return false;
		}

		if (m_points_.empty())
		{
			m_top_left_		= point;
			m_bottom_right_ = point;
		}
		else
		{
			m_top_left_.x		= std::min<float>(m_top_left_.x, point.x);
			m_top_left_.y		= std::min<float>(m_top_left_.y, point.y);
			m_bottom_right_.x	= std::max<float>(m_bottom_right_.x, point.x);
			m_bottom_right_.y	= std::max<float>(m_bottom_right_.y, point.y);
		}

        m_points_.push_back(point);
		return true;
    }

	bool BLOB::Add(const std::vector<cv::Point2f>& points)
	{
		bool inserted_all = true;
		m_points_.reserve(m_points_.size() + points.size());
		for (const cv::Point2f& point : points)
		{
			if (!Add(point))
			{
				inserted_all = false;
			}
		}

		return inserted_all;
	}

	void BLOB::UnsafeAdd(const cv::Point2f& point)
	{
		m_points_.push_back(point);
	}
	void BLOB::UnsafeAdd(const std::vector<cv::Point2f>& points)
	{
		m_points_.insert(m_points_.end(), points.begin(), points.end());
	}

	std::vector<cv::Point2f>& BLOB::GetPoints(void)
	{
		return m_points_;
	}

	const std::vector<cv::Point2f>& BLOB::GetPoints(void) const
	{
		return m_points_;
	}

	const cv::Point2f& BLOB::GetTopLeftPoint(void) const
	{
		return m_top_left_;
	}

	const cv::Point2f& BLOB::GetBottomRightPoint(void) const
	{
		return m_bottom_right_;
	}

	uint32_t BLOB::GetWidth(void) const
	{
		return static_cast<uint32_t>(std::sqrt(std::pow(m_top_left_.x - m_bottom_right_.x, 2)));
	}

	uint32_t BLOB::GetHeight(void) const
	{
		return static_cast<uint32_t>(std::sqrt(std::pow(m_top_left_.y - m_bottom_right_.y, 2)));
	}

	size_t BLOB::Size(void) const
	{
		return m_points_.size();
	}

	bool BLOB::BoxIntersectsWith(const BLOB& other) const
	{
		cv::Point2f other_top_left(other.GetTopLeftPoint());
		cv::Point2f other_bottom_right(other.GetBottomRightPoint());

		return BoxIntersectsWith(other_top_left, other_bottom_right);
	}

	bool BLOB::BoxIntersectsWith(const cv::Point2f& top_left, const cv::Point2f& bottom_right) const
	{
		return	m_bottom_right_.y	>= top_left.y &&
				m_top_left_.y		<= bottom_right.y &&
				m_bottom_right_.x	>= top_left.x &&
				m_top_left_.x		<= bottom_right.x;
	}
}