#ifndef __HOUGHTRANSFORM_TREEACCUMULATOR_H__
#define __HOUGHTRANSFORM_TREEACCUMULATOR_H__

#include "IAccumulator.h"
#include "LocationCell.h"

namespace HoughTransform
{
	/// <summary>
	///	An implementation of the IAccumulator interface using a tree-like structure for performance.
	/// </summary>
    class TreeAccumulator : public IAccumulator
    {
		public:
			/// <summary>
			/// Default constructor for the TreeAccumulator.
			/// </summary>
			/// <param name="axis_threshold">The threshold applied to axis values.</param>
			/// <param name="location_threshold">The threshold applied to location values.</param>
			/// <param name="count_threshold">The threshold applied to the count.</param>
			TreeAccumulator(const float axis_threshold, const float location_threshold, const size_t count_threshold);

			/// Adds an ellipse to the accumulator, this means finding a suitable location within
			/// the accumulator and averaging it with previously inserted ellipses.
			///	</summary>
			/// <param name="ellipse">The ellipse to be added and averaged.</param>
			/// <returns>The amount of ellipses that the added ellipse was averaged with + 1.</returns>
			size_t AddEllipse(Ellipse& ellipse);
			/// <summary>
			/// Clears the accumulator.
			/// </summary>
			void Clear(void);
			/// <summary>
			/// Accumulates all the ellipses in the accumulator that occur more than a certain threshold.
			/// </summary>
			/// <returns>A list of ellipses that has a count higher than a certain threshold.</returns>
			std::vector<Ellipse> Accumulate(void);

		private:
			float							m_axis_threshold_;
			float							m_location_threshold_;
			size_t							m_count_threshold_;
			std::forward_list<LocationCell> m_averaged_centers_;
    };
}
#endif // __HOUGHTRANSFORM_TREEACCUMULATOR_H__
