#include "MiscFunctionality.h"

#include <algorithm>
#include <random>

namespace ASAP::MiscFunctionality
{
	std::vector<uint32_t> CreateListOfRandomIntegers(size_t size)
	{
		std::vector<uint32_t> random_numbers(size);
		for (uint32_t element = 0; element < size; ++element)
		{
			random_numbers[element] = element;
		}
		std::shuffle(random_numbers.begin(), random_numbers.end(), std::random_device());
		return random_numbers;
	}
}