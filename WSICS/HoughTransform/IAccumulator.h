#ifndef __HOUGHTRANSFORM_IACCUMULATOR_H__
#define __HOUGHTRANSFORM_IACCUMULATOR_H__

#include <vector>

#include "Ellipse.h"

namespace WSICS::HoughTransform
{
	/// <summary>
	/// An interface for an accumulator, an implementor should be able to receive ellipses and
	///	average these ellipses upon an accumulation request.
	/// </summary>
    class IAccumulator
    {
		public:
			/// <summary>
			/// Destructs the object.
			/// </summary>
			~IAccumulator(void) { };

			/// <summary>
			/// Adds an ellipse to the accumulator, this means finding a suitable location within
			/// the accumulator and averaging it with previously inserted ellipses.
			///	</summary>
			/// <param name="ellipse">The ellipse to be added and averaged.</param>
			/// <returns>The amount of ellipses that the added ellipse was averaged with + 1.</returns>
			virtual size_t AddEllipse(Ellipse& ellipse)		= 0;
			/// <summary>
			/// Clears the accumulator.
			/// </summary>
			virtual void Clear(void)							= 0;
			/// <summary>
			/// Accumulates all the ellipses in the accumulator that occur more than a certain threshold.
			/// </summary>
			/// <returns>A list of ellipses that has a count higher than a certain threshold.</returns>
			virtual std::vector<Ellipse> Accumulate(void)	= 0;
    };
}
#endif // __HOUGHTRANSFORM_IACCUMULATOR_H__
