#ifndef __WSICS_IO_LOG_LEVEL__
#define __WSICS_IO_LOG_LEVEL__

#include <string>
#include <vector>

namespace WSICS::IO::Logging
{
	/// <summary>Represents the different levels of output for the program.</summary>
	enum LogLevel { SILENT, NORMAL, DEBUG };

	/// <summary>Translates the enumerable to a string.</summary>
	/// <param name="level">The log level to convert to string.</param>
	/// <returns>The enumerable as string.</returns>
	std::string GetLogLevelAsString(IO::Logging::LogLevel level);
	/// <summary>Translates the string into a log level.</summary>
	/// <param name="level">The string to convert to a log level.</param>
	/// <returns>The string as enumerable.</returns>
	LogLevel GetLogLevelFromString(const std::string& level);
	/// <summary>Returns the full list of enumerables converted to string.</summary>
	/// <returns>The complete list of converted enumerables.</returns>
	std::vector<std::string> GetLogLevelsAsString(void);
}
#endif // __WSICS_IO_LOG_LEVEL__