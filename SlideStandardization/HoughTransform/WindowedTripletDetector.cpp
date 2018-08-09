#include "WindowedTripletDetector.h"

#define _USE_MATH_DEFINES

#include <cmath>
#include <cstdlib> // rand
#include <math.h> // M_PI
#include <stdexcept>

namespace HoughTransform
{
    //******************************************************************************
    // Constructors / Destructors
    //******************************************************************************

	WindowedTripletDetector::WindowedTripletDetector(void) : parameters(GetStandardParameters()), m_blob_window_(this->parameters.window_size)
	{
	}

	WindowedTripletDetector::WindowedTripletDetector(const WindowedTripletDetectorParameters parameters) : parameters(parameters), m_blob_window_(this->parameters.window_size)
    {
    }

    //******************************************************************************
    // Public Member Functions
    //******************************************************************************

	void WindowedTripletDetector::Initialize(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const ASAP::Image_Processing::BLOB_Operations::MaskType mask_type)
    {
		m_blob_window_ = ASAP::Image_Processing::BLOB_Operations::BLOB_Window(m_blob_window_.GetWindowSize(), binary_matrix, output_matrix, mask_type);
		UpdateWindowInformation_();
    }

	void WindowedTripletDetector::Initialize(const cv::Mat& labeled_blob_matrix, const cv::Mat& stats_array)
	{
		m_blob_window_ = ASAP::Image_Processing::BLOB_Operations::BLOB_Window(m_blob_window_.GetWindowSize(), labeled_blob_matrix, stats_array);
		UpdateWindowInformation_();
	}

	bool WindowedTripletDetector::IsInitialized(void)
	{
		return !m_blob_window_.GetAllMatrixBLOBs().empty();
	}

    PointCollection WindowedTripletDetector::GetNextTriplet(void)
    {
		CheckValidAccess_();

		// If triplets can still be collected.
		if (Size() - m_deleted_points_.size() > 3)
		{
			std::pair<size_t, cv::Point2f*> origin(GetRandomLabeledPoint$());
			switch (this->parameters.point_selection)
			{
				case POINT_SELECTION_FULLY_LABELED:					return AcquireRandomTriplet_(true, true, origin);				break;
				case POINT_SELECTION_PARTIALLY_LABELED:				return AcquireRandomTriplet_(true, false, origin);				break;
				case POINT_SELECTION_FULLY_RANDOM:					return AcquireRandomTriplet_(false, false, origin);				break;
				case POINT_SELECTION_FULLY_LABELED_DIST_RESTR:		return AcquireRangeRestrictedTriplet_(true, true, origin);		break;
				case POINT_SELECTION_PARTIALLY_LABELED_DIST_RESTR:	return AcquireRangeRestrictedTriplet_(true, false, origin);		break;
				case POINT_SELECTION_FULLY_RANDOM_DIST_RESTR:		return AcquireRangeRestrictedTriplet_(false, false, origin);	break;
			}
		}
		return PointCollection();
    }

	void WindowedTripletDetector::Clear(void)
	{
		m_labeled_blobs_.clear();
		m_deleted_points_.clear();
		m_labeled_points_.clear();
		m_current_labels_.clear();
		m_blob_window_ = ASAP::Image_Processing::BLOB_Operations::BLOB_Window(m_blob_window_.GetWindowSize());
	}

	bool WindowedTripletDetector::Next(void)
	{
		CheckValidAccess_();

		bool shifted_window(m_blob_window_.ShiftWindowForward());
		if (!shifted_window)
		{
			return false;
		}

		UpdateWindowInformation_();

		return true;
	}

	void WindowedTripletDetector::Reset(void)
	{
		CheckValidAccess_();

		m_blob_window_.ShiftWindowToBegin();
		UpdateWindowInformation_();
	}

    void WindowedTripletDetector::Simplify(const HoughTransform::Ellipse& ellipse)
    {
		CheckValidAccess_();

		float sin_theta			= std::sin(ellipse.theta);
		float cos_theta			= std::cos(ellipse.theta);
		float outer_major_axis	= std::pow(ellipse.major_axis + this->parameters.ellipse_removal_range, 2);
		float outer_minor_axis	= std::pow(ellipse.minor_axis + this->parameters.ellipse_removal_range, 2);

		// Removes points from the BLOBs and any empty blob.
		std::vector<size_t> blobs_to_remove;
		for (std::pair<size_t, std::vector<cv::Point2f*>> labeled_points : m_labeled_points_)
		{
			labeled_points.second.erase(std::remove_if(labeled_points.second.begin(), labeled_points.second.end(),
				[sin_theta, cos_theta, outer_major_axis, outer_minor_axis, ellipse, this](cv::Point2f* point)
			{
				cv::Point2f adjusted_point(*point - ellipse.center);

				float a = std::pow(adjusted_point.y * sin_theta + adjusted_point.x * cos_theta, 2);
				float b = std::pow(adjusted_point.y * cos_theta - adjusted_point.x * sin_theta, 2);
				float outer_result = a / outer_major_axis + b / outer_minor_axis;

				if (outer_result < 1)
				{
					m_deleted_points_.insert(point);
					return true;
				}
				return false;
			}),
			labeled_points.second.end());

			if (labeled_points.second.empty())
			{
				blobs_to_remove.emplace_back(labeled_points.first);
			}
		}

		for (size_t blob_id : blobs_to_remove)
		{
			m_labeled_points_.erase(blob_id);
			m_current_labels_.erase(std::remove(m_current_labels_.begin(), m_current_labels_.end(), blob_id), m_current_labels_.end());
		}
    }

	size_t WindowedTripletDetector::Size(void)
	{
		return m_total_window_points_;
	}

    bool WindowedTripletDetector::Verify(const Ellipse& ellipse)
    {
        float major_axis = ellipse.major_axis;
		float minor_axis = ellipse.minor_axis;

        float estimated_pixels_on_ellipse = M_PI * (3.0f * (major_axis + minor_axis) - std::sqrt((3.0f * major_axis + minor_axis) * (major_axis + 3.0f * minor_axis)));

        int pixels_on_ellipse = 0;

        float sin_theta			= std::sin(ellipse.theta);
        float cos_theta			= std::cos(ellipse.theta);
        float inner_major_axis	= std::pow(major_axis - this->parameters.ellipse_range_tolerance, 2);
        float inner_minor_axis	= std::pow(minor_axis - this->parameters.ellipse_range_tolerance, 2);
        float outer_major_axis	= std::pow(major_axis + this->parameters.ellipse_range_tolerance, 2);
        float outer_minor_axis	= std::pow(minor_axis + this->parameters.ellipse_range_tolerance, 2);

		// Loops through the undeleted points.
		for (std::pair<size_t, std::vector<cv::Point2f*>> labeled_points : m_labeled_points_)
		{
			for (cv::Point2f* point : labeled_points.second)
			{
				cv::Point2f adjusted_point(*point - ellipse.center);

				float a				= std::pow(adjusted_point.y * sin_theta + adjusted_point.x * cos_theta, 2);
				float b				= std::pow(adjusted_point.y * cos_theta - adjusted_point.x * sin_theta, 2);
				float inner_result	= a / inner_major_axis + b / inner_minor_axis;
				float outer_result	= a / outer_major_axis + b / outer_minor_axis;

				if (inner_result > 1 && outer_result < 1)
				{
					pixels_on_ellipse++;
				}
			}
		}

		// Loops through the deleted points.
		for (cv::Point2f* point : m_deleted_points_)
		{
			cv::Point2f adjusted_point(*point - ellipse.center);

			float a				= std::pow(adjusted_point.y * sin_theta + adjusted_point.x * cos_theta, 2);
			float b				= std::pow(adjusted_point.y * cos_theta - adjusted_point.x * sin_theta, 2);
			float inner_result	= a / inner_major_axis + b / inner_minor_axis;
			float outer_result	= a / outer_major_axis + b / outer_minor_axis;

			if (inner_result > 1 && outer_result < 1)
			{
				pixels_on_ellipse++;
			}
		}

        return (static_cast<float>(pixels_on_ellipse) / static_cast<float>(estimated_pixels_on_ellipse) > this->parameters.min_coverage);
    }

	WindowedTripletDetectorParameters WindowedTripletDetector::GetStandardParameters(void)
	{
		return { 2.0f, 1.0f, 0.8f, 7.0f, 40.0f, 6.0f, 150, POINT_SELECTION_FULLY_LABELED_DIST_RESTR };
	}

	//******************************************************************************
	// Protected Member Functions
	//******************************************************************************

	inline std::pair<size_t, cv::Point2f*> WindowedTripletDetector::GetRandomLabeledPoint$(void)
	{
		size_t label = m_current_labels_[std::rand() % m_current_labels_.size()];
		return GetRandomLabeledPoint$(label);
	}

	inline std::pair<size_t, cv::Point2f*> WindowedTripletDetector::GetRandomLabeledPoint$(const size_t label)
	{
		std::vector<cv::Point2f*>& point_vector(m_labeled_points_[label]);
		return { label, point_vector[std::rand() % point_vector.size()] };
	}

	//******************************************************************************
	// Private Member Functions
	//******************************************************************************

	PointCollection WindowedTripletDetector::AcquireRandomTriplet_(const bool a_from_same_label, const bool b_from_same_label, std::pair<size_t, cv::Point2f*>& labeled_origin)
	{
		// Initializes the point vectors. Depending on the settings, one or both will be initialized.
		std::vector<std::pair<size_t, cv::Point2f*>> label_points_within_range;
		std::vector<std::pair<size_t, cv::Point2f*>> points_within_range;
		if (a_from_same_label || b_from_same_label)
		{
			label_points_within_range = std::move(GetPointsFromRadius_(*labeled_origin.second, labeled_origin.first));
		}
		if (!a_from_same_label || !b_from_same_label)
		{
			points_within_range = std::move(GetPointsFromRadius_(*labeled_origin.second));
		}

		// Acquires the Alpha and Bravo points randomly.
		std::pair<size_t, cv::Point2f*> point_a = a_from_same_label ? label_points_within_range[std::rand() & label_points_within_range.size()]
																	: points_within_range[std::rand() & points_within_range.size()];
		std::pair<size_t, cv::Point2f*> point_b = b_from_same_label ? label_points_within_range[std::rand() & label_points_within_range.size()]
																	: points_within_range[std::rand() & points_within_range.size()];

		PointCollection collection;
		collection.points.push_back({ *labeled_origin.second, CalculateTangent_(labeled_origin) });
		collection.points.push_back({ *point_a.second, CalculateTangent_(point_a) });
		collection.points.push_back({ *point_b.second, CalculateTangent_(point_b) });
		return collection;
	}

	PointCollection WindowedTripletDetector::AcquireRangeRestrictedTriplet_(const bool a_from_same_label, const bool b_from_same_label, std::pair<size_t, cv::Point2f*>& labeled_origin)
	{
		// Initializes the point vectors. Depending on the settings, one or both will be initialized.
		std::vector<std::pair<size_t, cv::Point2f*>> label_points_within_range;
		std::vector<std::pair<size_t, cv::Point2f*>> points_within_range;
		if (a_from_same_label || b_from_same_label)
		{
			label_points_within_range = std::move(GetPointsFromRadius_(*labeled_origin.second, labeled_origin.first));
		}
		if (!a_from_same_label || !b_from_same_label)
		{
			points_within_range = std::move(GetPointsFromRadius_(*labeled_origin.second));
		}

		// Leaves the points empty if the corresponding vectors are empty.
		std::pair<size_t, cv::Point2f*> point_a;
		std::pair<size_t, cv::Point2f*> point_b;

		// Acquires the Alpha point.
		if ((a_from_same_label && !label_points_within_range.empty()) || (!a_from_same_label && !points_within_range.empty()))
		{
			point_a = a_from_same_label ? label_points_within_range[std::rand() % label_points_within_range.size()] : points_within_range[std::rand() % points_within_range.size()];
		}

		// Filters the range around Alpha to ensure the remaining points are within the correct ranges and then attempts to acquire Bravo.
		if (!label_points_within_range.empty())
		{
			label_points_within_range.erase(std::remove_if(label_points_within_range.begin(), label_points_within_range.end(), [this, point_a](std::pair<size_t, cv::Point2f*>& labeled_point)
			{
				return !IsPositionedWithinRadius_(*point_a.second, *labeled_point.second, this->parameters.max_point_distance) ||
						IsPositionedWithinRadius_(*point_a.second, *labeled_point.second, this->parameters.min_point_distance);
			}),
			label_points_within_range.end());
		}
		if (!points_within_range.empty())
		{
			points_within_range.erase(std::remove_if(points_within_range.begin(), points_within_range.end(), [this, point_a](const std::pair<size_t, cv::Point2f*>& labeled_point)
			{
				return !IsPositionedWithinRadius_(*point_a.second, *labeled_point.second, this->parameters.max_point_distance) ||
					IsPositionedWithinRadius_(*point_a.second, *labeled_point.second, this->parameters.min_point_distance);
			}),
			points_within_range.end());
		}

		// Attempts to acquire Bravo within the range of Alpha.
		if ((b_from_same_label && !label_points_within_range.empty()) || (!b_from_same_label && !points_within_range.empty()))
		{
			point_b = b_from_same_label ? label_points_within_range[std::rand() % label_points_within_range.size()] : points_within_range[std::rand() % points_within_range.size()];
		}

		PointCollection collection;
		if (labeled_origin.second && point_a.second && point_b.second)
		{
			collection.points.push_back({ *labeled_origin.second, CalculateTangent_(labeled_origin) });
			collection.points.push_back({ *point_a.second, CalculateTangent_(point_a) });
			collection.points.push_back({ *point_b.second, CalculateTangent_(point_b) });
		}
		return collection;
	}

	Line WindowedTripletDetector::CalculateTangent_(std::pair<size_t, cv::Point2f*>& point)
	{
		std::vector<cv::Point2f> points;
		for (cv::Point2f& blob_point : m_labeled_blobs_[point.first]->GetPoints())
		{
			cv::Point2f difference(blob_point - *point.second);

			if (std::fabs(difference.x) < std::fabs(this->parameters.tangent_search_radius) && std::fabs(difference.y) < std::fabs(this->parameters.tangent_search_radius))
			{
				points.push_back(blob_point);
			}
			if (points.size() == static_cast<size_t>(this->parameters.tangent_search_radius * 2))
			{
				break;
			}
		}

		if (points.size() >= 2)
		{
			return Line(*point.second, points);
		}
		return Line();
	}

	void WindowedTripletDetector::CheckValidAccess_(void)
	{
		if (!IsInitialized())
		{
			throw std::runtime_error("No BLOB's are present within the WindowedTripletDetector.");
		}
	}

	std::vector<std::pair<size_t, cv::Point2f*>> WindowedTripletDetector::GetPointsFromRadius_(const cv::Point2f& center)
	{
		std::vector<std::pair<size_t, cv::Point2f*>> points_within_radius;
		for (std::pair<const size_t, std::vector<cv::Point2f*>>& labeled_points : m_labeled_points_)
		{
			for (cv::Point2f* point : labeled_points.second)
			{
				if (IsPositionedWithinRadius_(center, *point, this->parameters.max_point_distance) && !IsPositionedWithinRadius_(center, *point, this->parameters.min_point_distance))
				{
					points_within_radius.push_back({ labeled_points.first, point });
				}
			}
		}
		return points_within_radius;
	}

	std::vector<std::pair<size_t, cv::Point2f*>> WindowedTripletDetector::GetPointsFromRadius_(const cv::Point2f& center, const size_t label)
	{
		std::vector<std::pair<size_t, cv::Point2f*>> points_within_radius;
		for (cv::Point2f* point : m_labeled_points_[label])
		{
			if (IsPositionedWithinRadius_(center, *point, this->parameters.max_point_distance) && !IsPositionedWithinRadius_(center, *point, this->parameters.min_point_distance))
			{
				points_within_radius.push_back({ label, point });
			}
		}
		return points_within_radius;
	}

	// TODO: Move intoa geometry type namespace.
	bool WindowedTripletDetector::IsPositionedWithinRadius_(const cv::Point2f& center, cv::Point2f& position, const float radius)
	{
		return std::pow(center.x - position.x, 2) + std::pow(center.y - position.y, 2) < std::pow(radius, 2);
	}

	void WindowedTripletDetector::RemoveLabeledPoint_(std::pair<size_t, cv::Point2f*>& labeled_point)
	{
		m_deleted_points_.insert(labeled_point.second);

		std::vector<cv::Point2f*>& labeled_points_vector(m_labeled_points_.at(labeled_point.first));
		labeled_points_vector.erase(std::remove(labeled_points_vector.begin(), labeled_points_vector.end(), labeled_point.second), labeled_points_vector.end());

		if (labeled_points_vector.empty())
		{
			m_labeled_points_.erase(labeled_point.first);
			m_current_labels_.erase(std::remove(m_current_labels_.begin(), m_current_labels_.end(), labeled_point.first), m_current_labels_.end());
		}
	}

	void WindowedTripletDetector::UpdateWindowInformation_(void)
	{
		// Clears the containers that will need to rebuild their information.
		m_total_window_points_ = 0;
		m_deleted_points_.clear();
		m_labeled_points_.clear();
		m_current_labels_.clear();

		// Replaces the BLOB container.
		m_labeled_blobs_ = m_blob_window_.GetWindowBLOBs();

		// Filters BLOBs that don't contain the min amount of points to draw at least the this->parameters.min_point_distance between two points.
		for (auto it = m_labeled_blobs_.begin(); it != m_labeled_blobs_.end();)
		{
			if (it->second->Size() < this->parameters.min_point_distance)
			{
				it = m_labeled_blobs_.erase(it);
			}
			else
			{
				++it;
			}
		}

		// Initializes the list of labels, which is used by the randomize function to select a random point.
		m_labeled_points_.reserve(m_labeled_blobs_.size());
		m_current_labels_.reserve(m_labeled_blobs_.size());
		for (std::pair<const size_t, ASAP::Image_Processing::BLOB_Operations::BLOB*>& labeled_blob : m_labeled_blobs_)
		{
			m_labeled_points_.insert({ labeled_blob.first, std::vector<cv::Point2f*>() });
			m_current_labels_.push_back(labeled_blob.first);

			// Inserts pointers towards the blob points into the labeled vectors.
			std::vector<cv::Point2f>& blob_points(labeled_blob.second->GetPoints());
			m_labeled_points_[labeled_blob.first].reserve(blob_points.size());
			for (cv::Point2f& point : blob_points)
			{
				m_labeled_points_[labeled_blob.first].push_back(&point);
			}

			m_total_window_points_ += blob_points.size();
		}
	}
}