#ifndef __IO_COMMANDLINE_INTERFACE_H__
#define __IO_COMMANDLINE_INTERFACE_H__

//#include <boost/filesystem.hpp>

#include <boost/program_options.hpp>
#include <string>

#include "Logging/LogHandler.h"

namespace WSICS::IO
{
	/// <summary></summary>
	class CommandLineInterface
	{
		public:
			/// <summary>Default constructor.</summary>
			CommandLineInterface(void);

			/// <summary>Executes the program, calling both standard and module specific functionality.</summary>
			/// <param name="argc">The amount of arguments contained in the multidimensional char array.</param>
			/// <param name="argv">Multidimensional char array containing the parameters.</param>
			void Execute(int argc, char * argv[]);

		protected:
			/// <summary>Sets the log level for the LogHandler, defining its output.</summary>
			/// <param name="level">The output log level for the LogHandler.</param>
			void SetLogLevel$(const IO::Logging::LogLevel level);

			/// <summary>Exectues module specific functionality.</summary>
			/// <param name="variables">A map containing the command line variables.</param>
			virtual void ExecuteModuleFunctionality$(const boost::program_options::variables_map& variables) = 0;
			/// <summary>Adds module specific parameter options to the command line interface.</summary>
			/// <param name="options">A reference to the options_description object, that'll hold the full list of parameters.</param>
			virtual void AddModuleOptions$(boost::program_options::options_description& options) = 0;
			/// <summary>Performs any module specific preparations.</summary>
			virtual void Setup$(void) = 0;

		private:
			IO::Logging::LogHandler	m_log_handler_;

			/// <summary>Adds standard parameter options, present in all the command line tools.</summary>
			/// <param name="options">A reference to the options_description object, that'll hold the full list of parameters.</param>
			void AddStandardOptions_(boost::program_options::options_description& options);
			
			/// <summary>Exectues standard functionality</summary>
			/// <param name="argc">The amount of arguments passed through the command line.</param>
			/// <param name="variables">The map containing the command line variables.</param>
			/// <param name="options">The map containing the parameter options.</param>
			/// <returns>Whether or not the program should continue.</returns>
			bool ExecuteStandardFunctionality_(const int argc, const boost::program_options::variables_map& variables, const boost::program_options::options_description& options);

			/// <summary>Prints the parameter options to the command line screen.</summary>
			/// <param name="options">The map containing the parameter options.</param>
			void ListOptions_(const boost::program_options::options_description& options);
	};
}
#endif // __IO_COMMANDLINE_INTERFACE_H__
