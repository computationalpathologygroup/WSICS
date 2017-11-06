#include "RandomizedHoughTransform.h"

#define _USE_MATH_DEFINES

#include <cmath>
#include <math.h>
#include <opencv2/core/core.hpp>

namespace HoughTransform
{
    //******************************************************************************
    // Constructors / Destructors
    //******************************************************************************
	RandomizedHoughTransform::RandomizedHoughTransform(void)
		: parameters(GetStandardParameters()), m_triplet_detector_parameters_(WindowedTripletDetector::GetStandardParameters())
	{
	}

	RandomizedHoughTransform::RandomizedHoughTransform(RandomizedHoughTransformParameters parameters)
		: parameters(parameters), m_triplet_detector_parameters_(WindowedTripletDetector::GetStandardParameters())
	{
	}

	RandomizedHoughTransform::RandomizedHoughTransform(RandomizedHoughTransformParameters hough_transform_parameters, WindowedTripletDetectorParameters triplet_detector_parameters)
		: parameters(hough_transform_parameters), m_triplet_detector_parameters_(triplet_detector_parameters)
	{
	}

    //******************************************************************************
    // Public Member Functions
    //******************************************************************************

	std::vector<Ellipse> RandomizedHoughTransform::Execute(const cv::Mat& binary_matrix, cv::Mat& output_matrix, const ASAP::Image_Processing::BLOB_Operations::MaskType mask_type)
	{
		WindowedTripletDetector triplet_detector(m_triplet_detector_parameters_);
		triplet_detector.Initialize(binary_matrix, output_matrix, mask_type);
		return Execute(triplet_detector);
	}

	std::vector<Ellipse> RandomizedHoughTransform::Execute(const cv::Mat& labeled_matrix, const cv::Mat& stats_array)
	{
		WindowedTripletDetector triplet_detector(m_triplet_detector_parameters_);
		triplet_detector.Initialize(labeled_matrix, stats_array);
		return Execute(triplet_detector);
	}

    std::vector<Ellipse> RandomizedHoughTransform::Execute(WindowedTripletDetector& triplet_detector)
    {
        TreeAccumulator combiner(this->parameters.ellipse_radii_threshold*(int)(this->parameters.combine_threshold), this->parameters.ellipse_position_threshold*(int)(this->parameters.combine_threshold), 0);
		size_t shifts = 0;
		 ellipses_found = 0;
		 ellipses_verified = 0;
		bool repeat_epoch = true;
		while (repeat_epoch || triplet_detector.Next())
		{
			++shifts;
			repeat_epoch = false;
			Ellipse best_ellipse;
			size_t	best_ellipse_count = 0;

			TreeAccumulator accumulator(this->parameters.ellipse_radii_threshold, this->parameters.ellipse_position_threshold, this->parameters.count_threshold);
			for (size_t epoch = 0; epoch < triplet_detector.Size() * this->parameters.epoch_size; ++epoch)
			{
				std::pair<bool, Ellipse> ellipse_result(EllipseFromTriplet_(triplet_detector.GetNextTriplet()));

				if (ellipse_result.first)
				{
					++ellipses_found;
				}

				if (ellipse_result.first && ProcessDetectedEllipse_(ellipse_result.second, best_ellipse, best_ellipse_count, repeat_epoch, accumulator, combiner, triplet_detector))
				{
					epoch = 0;
				}
			}
		}

          return combiner.Accumulate();
    }

	RandomizedHoughTransformParameters RandomizedHoughTransform::GetStandardParameters(void)
	{
		return { 5, 4.0f, 8.0f, 1.0f, 5.0f, 25.0f, 30.0f, MIDPOINT_CALCULATION_OPTIMAL, ELLIPSE_REMOVAL_SIMPLE, COMBINE_THRESHOLD_FACTOR2, TANGENT_VERIFICATION_TRIPLE };
	}

	//******************************************************************************
	// Private Member Functions
	//******************************************************************************

	bool RandomizedHoughTransform::ComputeParameters_(PointCollection& triplet, Ellipse& ellipse)
	{
		double a_data[9];
		double b_data[] = { 1.0, 1.0, 1.0 };

		for (int i = 0; i < 3; i++)
		{
			a_data[i * 3 + 0] = triplet.points[i].first.x  * triplet.points[i].first.x;
			a_data[i * 3 + 1] = 2 * triplet.points[i].first.x  * triplet.points[i].first.y;
			a_data[i * 3 + 2] = triplet.points[i].first.y  * triplet.points[i].first.y;
		}

		cv::Mat matrixView = cv::Mat::zeros(3, 3, CV_32FC1);
		cv::Mat vectorView = cv::Mat::zeros(3, 1, CV_32FC1);
		cv::Mat dst = cv::Mat::zeros(3, 1, CV_32FC1);

		int counter = 0;
		for (int i = 0; i<3; ++i)
		{
			for (int j = 0; j<3; ++j)
			{
				matrixView.at<float>(i, j) = a_data[counter];
				++counter;
			}
		}

		vectorView.at<float>(0, 0) = 1; vectorView.at<float>(1, 0) = 1; vectorView.at<float>(2, 0) = 1;

		cv::solve(matrixView, vectorView, dst, 1);

		float a = (ellipse.theta		= dst.at<float>(0, 0));
		float b = (ellipse.major_axis	= dst.at<float>(1, 0));
		float c = (ellipse.minor_axis	= dst.at<float>(2, 0));

		if (a*c - pow(b, 2) <= 0)
		{
			return false;
		}
		else if (_isnan(a) || _isnan(b) || _isnan(c))
		{
			return false;
		}
		return true;
	}

	void RandomizedHoughTransform::ConvertParameters_(Ellipse& ellipse) const
	{
		float new_theta, new_major_axis, new_minor_axis;
		if (ellipse.theta == ellipse.minor_axis)
		{
			new_theta = 0;
		}
		else
		{
			new_theta = 0.5 * std::atan(2 * ellipse.major_axis / (ellipse.theta - ellipse.minor_axis));
			if (ellipse.theta > ellipse.minor_axis)
			{
				if (ellipse.major_axis < 0)
				{
					new_theta = new_theta - (-1) * 0.5 * M_PI;
				}
				else if (ellipse.major_axis > 0)
				{
					new_theta = new_theta - (1) * 0.5 * M_PI;
				}
			}
		}

		new_major_axis = std::sqrt(std::cos(2 * new_theta) / (ellipse.theta - (ellipse.theta + ellipse.minor_axis) * std::pow(std::sin(new_theta), 2)));
		new_minor_axis = new_major_axis / std::sqrt((ellipse.theta + ellipse.minor_axis) * std::pow(new_major_axis, 2) - 1);

		ellipse.theta		= new_theta;
		ellipse.major_axis	= new_major_axis;
		ellipse.minor_axis	= new_minor_axis;
	}

	cv::Point2f RandomizedHoughTransform::DetermineCenter_(PointCollection& triplet)
	{
		if (triplet.points[0].second.IsParallelWith(triplet.points[1].second))
		{
			return (triplet.points[0].first + triplet.points[1].first) / 2.0f;
		}
		else if (triplet.points[1].second.IsParallelWith(triplet.points[2].second))
		{
			return (triplet.points[1].first + triplet.points[2].first) / 2.0f;
		}
		else
		{
			switch (this->parameters.midpoint_calculation)
			{
				case(MIDPOINT_CALCULATION_DEFAULT):
				{
					cv::Point2f intersect_AB = triplet.points[0].second.Intersect(triplet.points[1].second);
					cv::Point2f intersect_BC = triplet.points[1].second.Intersect(triplet.points[2].second);
					cv::Point2f midpoint_AB = (triplet.points[0].first + triplet.points[1].first) / 2.0f;
					cv::Point2f midpoint_BC = (triplet.points[1].first + triplet.points[2].first) / 2.0f;

					Line line_AB = Line(intersect_AB, midpoint_AB);
					Line line_BC = Line(intersect_BC, midpoint_BC);
					return line_AB.Intersect(line_BC);
					break;
				}
				case(MIDPOINT_CALCULATION_OPTIMAL):
				{
					cv::Point2f intersect_AB = triplet.points[0].second.Intersect(triplet.points[1].second);
					cv::Point2f intersect_BC = triplet.points[1].second.Intersect(triplet.points[2].second);
					cv::Point2f intersect_CA = triplet.points[2].second.Intersect(triplet.points[0].second);

					cv::Point2f midpoint_AB = (triplet.points[0].first + triplet.points[1].first) / 2.0f;
					cv::Point2f midpoint_BC = (triplet.points[1].first + triplet.points[2].first) / 2.0f;
					cv::Point2f midpoint_CA = (triplet.points[2].first + triplet.points[0].first) / 2.0f;

					float dist_AB = std::pow(intersect_AB.x - midpoint_AB.x, 2) + std::pow(intersect_AB.y - midpoint_AB.y, 2);
					float dist_BC = std::pow(intersect_BC.x - midpoint_BC.x, 2) + std::pow(intersect_BC.y - midpoint_BC.y, 2);
					float dist_CA = std::pow(intersect_CA.x - midpoint_CA.x, 2) + std::pow(intersect_CA.y - midpoint_CA.y, 2);

					if (dist_AB < dist_BC && dist_AB < dist_CA)
					{
						Line line_CA = Line(intersect_CA, midpoint_CA);
						Line line_BC = Line(intersect_BC, midpoint_BC);
						return line_CA.Intersect(line_BC);
					}
					else if (dist_BC < dist_AB && dist_BC < dist_CA)
					{
						Line line_CA = Line(intersect_CA, midpoint_CA);
						Line line_AB = Line(intersect_AB, midpoint_AB);
						return line_CA.Intersect(line_AB);
					}
					else
					{
						Line line_AB = Line(intersect_AB, midpoint_AB);
						Line line_BC = Line(intersect_BC, midpoint_BC);
						return line_AB.Intersect(line_BC);
					}
					break;
				}
			}
		}

		return cv::Point2f(0, 0);
	}

	bool RandomizedHoughTransform::HasCorrectRadius_(const Ellipse& ellipse) const
	{
		return ellipse.major_axis <= this->parameters.max_ellipse_radius && ellipse.minor_axis >= this->parameters.min_ellipse_radius;
	}

	bool RandomizedHoughTransform::FitsContours_(PointCollection& triplet, const Ellipse& ellipse) const
	{
		char valid = 0;
		for (std::pair<cv::Point2f, Line>& triplet_pair : triplet.points)
		{
			float difference = std::fabs(ellipse.GetTangent(triplet_pair.first) - triplet_pair.second.GetAngle());

			if (difference > 0.5f * M_PI)
			{
				difference = M_PI - difference;
			}

			if (difference * 180 / M_PI < this->parameters.tangent_tolerance)
			{
				++valid;
			}
		}
		
		return valid >= this->parameters.tangent_verification;
	}

	std::pair<bool, Ellipse> RandomizedHoughTransform::EllipseFromTriplet_(PointCollection& triplet)
	{
		Ellipse ellipse;
		bool valid_ellipse = false;

		if (triplet.points.size() >= 3)
		{
			// Compute parameters using the triplet.
			ellipse = Ellipse(DetermineCenter_(triplet), 0, 0, 0);
			triplet -= ellipse.center;

			// Use the tangents from the image and compare those with the tangents calculated from the parameters.
			if (ComputeParameters_(triplet, ellipse) && FitsContours_(triplet, ellipse))
			{
				// filter the ellipse based on the calculated radius.
				ConvertParameters_(ellipse);
				valid_ellipse = HasCorrectRadius_(ellipse);
			}
		}

		return { valid_ellipse, ellipse };
	}

	bool RandomizedHoughTransform::ProcessDetectedEllipse_(Ellipse& ellipse,
		Ellipse& best_ellipse,
		size_t& best_ellipse_count,
		bool& repeat_epoch,
		TreeAccumulator& accumulator,
		TreeAccumulator& combiner,
		WindowedTripletDetector& triplet_detector)
	{
		bool reset_epoch = false;

		// Add the ellipse to the accumulator, averaging it if collisions are found.
		size_t count = accumulator.AddEllipse(ellipse);

		// When using point deletion, an ellipse its points will be deleted from the current window if
		// they fit certain criteria. This can result in restarting the epoch for the current window
		// and the clearing of the accumulator if "EMPTY_ACCUM" point deletion method is set.
		if (this->parameters.ellipse_removal_method == ELLIPSE_REMOVAL_BEST_IN_EPOCH)
		{
			if (count > best_ellipse_count)
			{
				best_ellipse = ellipse;
				best_ellipse_count = count;
			}

			if (best_ellipse_count > this->parameters.count_threshold)
			{
				if (triplet_detector.Verify(best_ellipse))
				{
					combiner.AddEllipse(best_ellipse);
				}
				triplet_detector.Simplify(best_ellipse);
				repeat_epoch = true;
			}
		}
		else if (count == this->parameters.count_threshold)
		{
			if ((this->parameters.ellipse_removal_method == ELLIPSE_REMOVAL_SIMPLE || this->parameters.ellipse_removal_method == ELLIPSE_REMOVAL_EMPTY_ACCUMULATOR) && triplet_detector.Verify(ellipse))
			{
				//triplet_detector.Simplify(ellipse);
				combiner.AddEllipse(ellipse);
				++ellipses_verified;
				if (this->parameters.ellipse_removal_method == ELLIPSE_REMOVAL_EMPTY_ACCUMULATOR)
				{
					reset_epoch = true;
					accumulator.Clear();
				}
			}
		}

		if (this->parameters.ellipse_removal_method == ELLIPSE_REMOVAL_NONE || this->parameters.ellipse_removal_method == ELLIPSE_REMOVAL_ALL_IN_EPOCH)
		{
			std::vector<Ellipse> ellipses(accumulator.Accumulate());
			for (Ellipse& accumulated_ellipse : ellipses)
			{
				if (triplet_detector.Verify(accumulated_ellipse))
				{
					if (this->parameters.ellipse_removal_method == ELLIPSE_REMOVAL_ALL_IN_EPOCH)
					{
						triplet_detector.Simplify(accumulated_ellipse);
						repeat_epoch = true;
					}

					combiner.AddEllipse(accumulated_ellipse);
				}
			}
		}

		return reset_epoch;
	}
}