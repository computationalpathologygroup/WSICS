#include <cmath>

#define _USE_MATH_DEFINES

#include <math.h>

#include "AveragedEllipseParameters.h"

namespace WSICS::HoughTransform
{
	AveragedEllipseParameters::AveragedEllipseParameters(const Ellipse& ellipse) : m_count_(1)
    {
		m_averaged_information_ = { ellipse.major_axis, ellipse.minor_axis, ellipse.theta };
    }

    size_t AveragedEllipseParameters::Add(Ellipse& ellipse)
    {
		// Can't directly average theta, as there's a turn over.
        if(std::fabs(ellipse.theta - m_averaged_information_.theta) > 0.5 * M_PI)
        {
            if(m_averaged_information_.theta > ellipse.theta)
            {
				m_averaged_information_.theta -= M_PI;
            }
            else
            {
				ellipse.theta -= M_PI;
            }
        }
		m_averaged_information_.theta = (m_averaged_information_.theta * m_count_ + ellipse.theta) / static_cast<float>(m_count_ + 1);

        // Returns theta to the correct range.
        if(m_averaged_information_.theta < -0.5 * M_PI)
        {
			m_averaged_information_.theta += M_PI;
        }

		m_averaged_information_.major_axis = (m_averaged_information_.major_axis * m_count_ + ellipse.major_axis) / static_cast<float>(m_count_ + 1);
		m_averaged_information_.minor_axis = (m_averaged_information_.minor_axis * m_count_ + ellipse.minor_axis) / static_cast<float>(m_count_ + 1);

		ellipse.theta		= m_averaged_information_.theta;
		ellipse.major_axis	= m_averaged_information_.major_axis;
		ellipse.minor_axis	= m_averaged_information_.minor_axis;

		m_count_++;
        return m_count_;
    }

	size_t AveragedEllipseParameters::GetCount(void) const
	{
		return m_count_;
	}
	
	AveragedEllipseParameters::AveragedEllipseInformation AveragedEllipseParameters::GetEllipseInformation(void) const
	{
		return m_averaged_information_;
	}
}