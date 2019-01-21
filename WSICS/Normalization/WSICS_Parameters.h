#ifndef __WSICS_NORMALIZATION_WSICSPARAMETERS__
#define __WSICS_NORMALIZATION_WSICSPARAMETERS__

#include <cstdint>

namespace WSICS::Normalization
{
	/// <summary>
	/// Holds all the normalization parameters that can be used to tune the WSICS algorithm execution.
	/// </summary>
	struct WSICS_Parameters
	{
		int32_t		minimum_ellipses;
		uint32_t	min_training_size;
		uint32_t	max_training_size;
		uint64_t	seed;
		float		hema_percentile;
		float		eosin_percentile;
		float		background_threshold;
		bool		consider_ink;
	};
}
#endif // __WSICS_NORMALIZATION_WSICSPARAMETERS__