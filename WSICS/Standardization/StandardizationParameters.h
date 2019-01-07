#pragma once
#include <cstdint>

namespace WSICS::Standardization
{
	/// <summary>
	/// Holds all the Standardization parameters that can be defined through the CLI.
	/// </summary>
	struct StandardizationParameters
	{
		uint32_t	minimum_ellipses;
		uint32_t	min_training_size;
		uint32_t	max_training_size;
		uint64_t	seed;
		float		hema_percentile;
		float		eosin_percentile;
		float		background_threshold;
		bool		consider_ink;
	};
}