#include "Random.h"

#include <algorithm>
#include <random>

namespace WSICS::Misc::Random
{
	std::vector<size_t> CreateListOfRandomIntegers(const size_t size)
	{
		std::vector<size_t> random_numbers(size);
		for (size_t element = 0; element < size; ++element)
		{
			random_numbers[element] = element;
		}
		std::shuffle(random_numbers.begin(), random_numbers.end(), std::random_device());
		return random_numbers;
	}

	std::vector<size_t> CreateListOfRandomIntegers(const size_t size, boost::mt19937_64& generator)
	{
		std::vector<size_t> random_numbers(size);
		for (size_t element = 0; element < size; ++element)
		{
			random_numbers[element] = element;
		}
		std::shuffle(random_numbers.begin(), random_numbers.end(), generator);
		return random_numbers;
	}
}