#include "LogLevel.h"

#include <stdexcept>

namespace WSICS::IO::Logging
{
	std::string GetLogLevelAsString(LogLevel level)
	{
		std::string level_as_string;
		switch (level)
		{
		case	 SILENT:	level_as_string = "silent";	 break;
		case	 NORMAL:	level_as_string = "normal";	 break;
		case	 DEBUG:		level_as_string = "debug";	 break;
		}
		return level_as_string;
	}

	LogLevel GetLogLevelFromString(const std::string& level)
	{
		std::vector<std::string> log_levels(GetLogLevelsAsString());
		for (size_t type = 0; type < log_levels.size(); ++type)
		{
			if (level == log_levels[type])
			{
				return LogLevel(type);
			}
		}

		throw std::runtime_error("Unable to ascertain log level.");
	}

	std::vector<std::string> GetLogLevelsAsString(void)
	{
		return std::vector<std::string>{ "silent", "normal", "debug" };
	}
}