#ifndef __WSICS_MISC_RANDOM__
#define __WSICS_MISC_RANDOM__

#include <stddef.h>
#include <vector>

#include <boost/random/mersenne_twister.hpp>

namespace WSICS::Misc::Random
{
	std::vector<size_t> CreateListOfRandomIntegers(const size_t size);
	std::vector<size_t> CreateListOfRandomIntegers(const size_t size, boost::mt19937_64& generator);
}
#endif // __WSICS_MISC_RANDOM__