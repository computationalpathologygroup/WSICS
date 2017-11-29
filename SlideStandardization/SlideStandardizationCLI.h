#ifndef __SLIDESTANDARDIZATION_CLI_H__
#define __SLIDESTANDARDIZATION_CLI_H__

#include <boost/filesystem.hpp>

#include "IO/CommandLineInterface.h"

/// <summary>
/// Defines the CLI interaction required to use the Slide Standardization libraries.
/// </summary>
class SlideStandardizationCLI : public IO::CommandLineInterface
{
	public:
		/// <summary>
		/// Default constructor.
		/// </summary>
		SlideStandardizationCLI(void);

	protected:
		/// <summary>
		/// Exectues module specific functionality.
		/// </summary>
		/// <param name="variables">A map containing the command line variables.</param>
		void ExecuteModuleFunctionality$(const boost::program_options::variables_map& variables);
		/// <summary>
		/// Adds module specific parameter options to the command line interface.
		/// </summary>
		/// <param name="options">A reference to the options_description object, that'll hold the full list of parameters.</param>
		void AddModuleOptions$(boost::program_options::options_description& options);
		/// <summary>
		/// Performs any module specific preparations.
		/// </summary>
		void Setup$(void);

	private:
		/// <summary>
		/// Takes the commandline input and checks its validity, passing the results into the references.
		/// </summary>
		/// <param name="variables">The variables gathered from the command line.</param>
		void AcquireAndSanitizeInput_(
			const boost::program_options::variables_map& variables,
			std::vector<boost::filesystem::path>& files_to_process,
			uint32_t& max_training_size,
			uint32_t& min_training_size,
			std::string& prefix,
			std::string& postfix,
			boost::filesystem::path& output_path,
			boost::filesystem::path& template_input,
			boost::filesystem::path& template_output,
			boost::filesystem::path& debug_dir,
			bool& contains_ink,
			bool& input_is_directory);

		void CreateDirectories_(
			const boost::filesystem::path& output_path,
			const boost::filesystem::path& template_output,
			const boost::filesystem::path& debug_directory,
			const std::vector<boost::filesystem::path>& files,
			const bool input_is_directory);

		/// <summary>
		/// Checks if the passed input parameter is a file or a directory, it then fills
		/// the vector with all the files that are eligble for processing.
		/// </summary>
		/// <param name="input_path">A path pointing either towards an image file, or a directory containing image files.</param>
		/// <returns>A vector holding all the eligble image files.</returns>
		std::vector<boost::filesystem::path> GatherImageFilenames_(const boost::filesystem::path input_path);
};
#endif // __SLIDESTANDARDIZATION_CLI_H__
