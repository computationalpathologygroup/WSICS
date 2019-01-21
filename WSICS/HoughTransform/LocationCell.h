#ifndef __WSICS_HOUGHTRANSFORM_LOCATIONCELL__
#define __WSICS_HOUGHTRANSFORM_LOCATIONCELL__

#include <vector>

#include "Ellipse.h"
#include "AveragedEllipseParameters.h"

namespace WSICS::HoughTransform
{
	/// <summary>
	/// A container/node class for the TreeAccumulator class containing an ellipse it's location
	/// and a forward linked list that consist of AveragedEllipseParameters.
	/// </summary>
    class LocationCell
    {
		public:
			/// <summary>
			/// The default constructor for the LocationCell.
			/// </summary>
			/// <param name="ellipse">The initial ellipse to add.</param>
			LocationCell(const Ellipse& ellipse);

			/// <summary>
			/// Adds an ellipse to this cell, and averages it's parameters with the existing one.
			/// </summary>
			/// <param name="ellipse">The ellipse to be added.</param>
			/// <param name="threshold">The threshold used for determining which averaged ellipse the parameters go in.</param>
			/// <returns>The new count of the averaged ellipse  that the parameters where added to.</returns>
			size_t Add(Ellipse& ellipse, const float threshold);

			/// <summary>
			/// Returns a reference to the linked list of averaged ellipse parameters.
			/// </summary>
			/// <returns>A reference to the linked list of averaged ellipse parameters.</returns>
			const std::vector<AveragedEllipseParameters>& GetAveragedEllipseParameterss(void) const;
			/// <summary>
			/// Returns the averaged ellipse parameters with the highest count.
			/// </summary>
			/// <returns>The averaged ellipse parameters with the highest count.</returns>
			const AveragedEllipseParameters& GetBestAveragedEllipseParameters(void) const;
			/// <summary>
			/// Returns the amount of added ellipses.
			/// </summary>
			/// <returns>The amount of added ellipses.</returns>
			size_t GetCount(void) const;
			/// <summary>
			/// Returns the averaged center of the ellipses.
			/// </summary>
			/// <returns>The averaged center of the ellipses.</returns>
			const cv::Point2f& GetCenter(void) const;

		private:
			std::vector<AveragedEllipseParameters>	m_averaged_ellipses_;
			AveragedEllipseParameters				m_best_averaged_ellipse_;
			cv::Point2f								m_center_;
			size_t									m_count_;
    };
}
#endif // __WSICS_HOUGHTRANSFORM_LOCATIONCELL__
