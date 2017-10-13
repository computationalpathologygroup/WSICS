#include "LocationCell.h"

#include <cmath>

namespace HoughTransform
{
    LocationCell::LocationCell(const Ellipse& ellipse) : m_best_averaged_ellipse_(nullptr), m_center_(ellipse.center), m_count_(1)
    {
		m_averaged_ellipses_.push_front(AveragedEllipseParameters(ellipse));
		m_best_averaged_ellipse_ = &*m_averaged_ellipses_.begin();
    }

	size_t LocationCell::Add(Ellipse& ellipse, const float threshold)
    {
		m_center_.x = (m_center_.x * m_count_ + ellipse.center.x) / static_cast<float>(m_count_ + 1);
		m_center_.y = (m_center_.y * m_count_ + ellipse.center.y) / static_cast<float>(m_count_ + 1);

		ellipse.center = m_center_;

		++m_count_;

		for (AveragedEllipseParameters& averaged_ellipse : m_averaged_ellipses_)
		{
			AveragedEllipseParameters::AveragedEllipseInformation averaged_ellipse_info = averaged_ellipse.GetEllipseInformation();

			float difference_major_axis = ellipse.major_axis - averaged_ellipse_info.major_axis;
			float difference_minor_axis = ellipse.minor_axis - averaged_ellipse_info.minor_axis;

			if (std::fabs(difference_major_axis) < threshold && std::fabs(difference_minor_axis))
			{
				size_t count = averaged_ellipse.Add(ellipse);
				if (averaged_ellipse.GetCount() > m_best_averaged_ellipse_->GetCount())
				{
					m_best_averaged_ellipse_ = &averaged_ellipse;
				}
				return count;
			}
		}

		m_averaged_ellipses_.push_front(AveragedEllipseParameters(ellipse));
		return 1;
    }

	const std::forward_list<AveragedEllipseParameters>& LocationCell::GetAveragedEllipseParameterss(void) const
	{
		return m_averaged_ellipses_;
	}

	const AveragedEllipseParameters& LocationCell::GetBestAveragedEllipseParameters(void) const
	{
		return *m_best_averaged_ellipse_;
	}

	size_t LocationCell::GetCount(void) const
	{
		return m_count_;
	}

	const cv::Point2f& LocationCell::GetCenter(void) const
	{
		return m_center_;
	}
}