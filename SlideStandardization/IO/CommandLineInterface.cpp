#include "CommandLineInterface.h"

#include "Logging/LogLevel.h"

namespace IO
{
	CommandLineInterface::CommandLineInterface(void) : m_log_handler_(IO::Logging::NORMAL)
	{
	}

	void CommandLineInterface::Execute(int argc, char * argv[])
	{
		// Registers and starts the log handlers prcoessing.
		IO::Logging::LogHandler::Register(&m_log_handler_);
		IO::Logging::LogHandler::GetInstance()->Initialize();

		boost::program_options::options_description options;
		AddStandardOptions_(options);
		AddModuleOptions$(options);

		bool encountered_error = false;

		boost::program_options::variables_map variables_map;
		try
		{
			boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(options).run(), variables_map);
			boost::program_options::notify(variables_map);
		}
		catch (...)
		{
			ListOptions_(options);
			encountered_error = true;
		}
		
		if (!encountered_error)
		{
			ExecuteStandardFunctionality_(argc, variables_map, options);
			ExecuteModuleFunctionality$(variables_map);
		}
	}

	void CommandLineInterface::SetLogLevel$(const IO::Logging::LogLevel level)
	{
		m_log_handler_.SetOutputLevel(level);
	}

	void CommandLineInterface::AddStandardOptions_(boost::program_options::options_description& options)
	{
		// Acquires the log levels as string.
		std::vector<std::string> log_levels(IO::Logging::GetLogLevelsAsString());
		std::string appended_log_levels;
		for (std::string& level : log_levels)
		{
			appended_log_levels += level + ", ";
		}
		appended_log_levels = appended_log_levels.substr(0, appended_log_levels.size() - 2);

		options.add_options()
			("help,h", boost::program_options::value<bool>()->default_value(false)->implicit_value(true), "Shows the usage of the program, as well as the available input parameters.")
			("log_level,l", boost::program_options::value<std::string>(), std::string("Sets the amount of log output generated. Options are: " + appended_log_levels).c_str());
	}

	void CommandLineInterface::ExecuteStandardFunctionality_(const int argc, const boost::program_options::variables_map& variables, const boost::program_options::options_description& options)
	{
		if (argc == 2 || variables["help"].as<bool>())
		{
			ListOptions_(options);
		}

		if (!variables["log_level"].empty())
		{
			std::string log_level(variables["log_level"].as<std::string>());
			std::transform(log_level.begin(), log_level.end(), log_level.begin(), ::tolower);

			SetLogLevel$(IO::Logging::GetLogLevelFromString(log_level));
		}
	}

	void CommandLineInterface::ListOptions_(const boost::program_options::options_description& options)
	{
		options.print(std::cout);
	}
}