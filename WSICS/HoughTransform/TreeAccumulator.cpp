#include "TreeAccumulator.h"

#include <cmath>

namespace WSICS::HoughTransform
{
    TreeAccumulator::TreeAccumulator(const float axis_threshold, const float location_threshold, const size_t count_threshold)
		: m_axis_threshold_(axis_threshold), m_location_threshold_(location_threshold), m_count_threshold_(count_threshold)
    {
    }

    void TreeAccumulator::Clear()
    {
		m_averaged_centers_.clear();
    }
	
    size_t TreeAccumulator::AddEllipse(Ellipse& ellipse)
    {
        if(std::isnan(ellipse.theta) || std::isnan(ellipse.major_axis) || std::isnan(ellipse.minor_axis))
        {
            return 0;
        }

		if (m_averaged_centers_.empty())
		{
			m_averaged_centers_.push_back(LocationCell(ellipse));
			return 1;
		}

		for (LocationCell& cell : m_averaged_centers_)
		{
			float x = ellipse.center.x - cell.GetCenter().x;
			float y = ellipse.center.y - cell.GetCenter().y;

			if (std::pow(x, 2) + std::pow(y, 2) < std::pow(m_location_threshold_, 2))
			{
				return cell.Add(ellipse, m_axis_threshold_);
			}
		}

		m_averaged_centers_.push_back(LocationCell(ellipse));
		return 1;
    }

	std::vector<Ellipse> TreeAccumulator::Accumulate()
    {
		std::vector<Ellipse> ellipses;

		for (LocationCell& cell : m_averaged_centers_)
		{
			if (cell.GetBestAveragedEllipseParameters().GetCount() >= m_count_threshold_)
			{
				AveragedEllipseParameters::AveragedEllipseInformation subnode_parameters(cell.GetBestAveragedEllipseParameters().GetEllipseInformation());
				ellipses.push_back(Ellipse(cell.GetCenter(), subnode_parameters.major_axis, subnode_parameters.minor_axis, subnode_parameters.theta));
			}
		}

		return ellipses;
    }
}