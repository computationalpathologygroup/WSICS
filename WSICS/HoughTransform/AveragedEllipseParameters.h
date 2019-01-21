#ifndef __WSICS_HOUGHTRANSFORM_PARAMETERCELL__
#define __WSICS_HOUGHTRANSFORM_PARAMETERCELL__

#include "Ellipse.h"

namespace WSICS::HoughTransform
{
	/// <summary>
	/// A container/node class for the TreeAccumulator class containing an ellipse its parameters.
	/// </summary>
    class AveragedEllipseParameters
    {
    public:
		/// <summary>
		/// Holds the information of the averaged ellipse parameters.
		/// </summary>
		struct AveragedEllipseInformation
		{
			float major_axis;
			float minor_axis;
			float theta;
		};

		/// <summary>
		/// Default constructor.
		/// </summary>
		/// <param name="ellipse">The ellipse to incorperate into the averaged whole.</param>
		AveragedEllipseParameters(const Ellipse& ellipse);

		/// <summary>
		/// Adds an ellipse to the whole, resulting in a weighted average.
		/// </summary>
		/// <param name="ellipse">The ellipse to incorperate into the averaged whole.</param>
		/// <returns>The amount of ellipses averaged.</returns>
        size_t Add(Ellipse& ellipse);
		/// <summary>
		/// Returns the amount of averaged ellipses.
		/// </summary>
		/// <returns>The amount of averaged ellipses.</returns>
		size_t GetCount(void) const;
		/// <summary>
		/// Returns the averaged ellipse information.
		/// </summary>
		/// <returns>The averaged ellipse information.</returns>
		AveragedEllipseInformation GetEllipseInformation(void) const;

	private:
		size_t						m_count_;
		AveragedEllipseInformation	m_averaged_information_;
    };
}
#endif // __WSICS_HOUGHTRANSFORM_PARAMETERCELL__