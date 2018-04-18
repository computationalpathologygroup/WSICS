#pragma once
#include <cstdint>

struct StandardizationParameters
{
	int32_t minimum_ellipses;
	uint32_t min_training_size;
	uint32_t max_training_size;
	float hema_percentile;
	float eosin_percentile;
	float background_threshold;
	bool consider_ink;
};