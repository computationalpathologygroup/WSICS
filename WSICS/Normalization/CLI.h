#ifndef __WSICS_STANDARDIZATION_CLI_H__
#define __WSICS_STANDARDIZATION_CLI_H__

#include <boost/filesystem.hpp>

#include "WSICS_Algorithm.h"
#include "../IO/CommandLineInterface.h"

namespace WSICS::Normalization
{
	/// <summary>
	/// Defines the CLI interaction required to use the Slide Standardization libraries.
	/// </summary>
	class CLI : public IO::CommandLineInterface
	{
		public:
			/// <summary>
			/// Default constructor.
			/// </summary>
			CLI(void);

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
			/// <param name="variables">The collection of parameters acquired from the CLI.</param>
			/// <param name="parameters">A struct containing all the exposed parameters.</param>
			/// <param name="files_to_process">A list of the files that require processing.</param>
			/// <param name="prefix">The prefix for a file, incase a directory has been offered.</param>
			/// <param name="postfix">The postfix for a file, incase a directory has been offered.</param>
			/// <param name="image_output">The file or directory path to where the image output should occur.</param>
			/// <param name="lut_output">The file or directory path to where the LUT output should occur.</param>
			/// <param name="template_input">A filepath to the template used for normalising the image.</param>
			/// <param name="template_output">The file or directory path to where the template output should occur.</param>
			/// <param name="debug_dir">The directory where debug data should be written to.</param>
			/// <param name="input_is_directory">Whether or not a file or directory path has been offered.</param>
			void AcquireAndSanitizeInput_(
				const boost::program_options::variables_map& variables,
				WSICS_Parameters& parameters,
				std::vector<boost::filesystem::path>& files_to_process,
				std::string& prefix,
				std::string& postfix,
				boost::filesystem::path& image_output,
				boost::filesystem::path& lut_output,
				boost::filesystem::path& template_input,
				boost::filesystem::path& template_output,
				boost::filesystem::path& debug_dir,
				bool& input_is_directory);


			/// <summary>
			/// Creates the directories required for outputting results and debug data.
			/// </summary>
			/// <param name="image_output">The directory path to where the image should be written.</param>
			/// <param name="lut_output">The directory path to where the lut table should be written.</param>
			/// <param name="template_output">The directory path to where the template should be written.</param>
			/// <param name="debug_directory">The directory path to where the debug data should be written.</param>
			/// <param name="files">The list of files that need to be processed.</param>
			/// <param name="input_is_directory">Whether or not the input parameter contains a directory path.</param>
			void CreateDirectories_(
				const boost::filesystem::path& image_output,
				const boost::filesystem::path& lut_output,
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


			/// <summary>
			/// Sets the complete output path for whatever filepath is provided.
			/// </summary>
			/// <param name="path">The original filepath.</param>
			/// <param name="extension">What extension the file should have.</param>
			/// <param name="filename">The filename appended to the path.</param>
			/// <returns>The definitive output path for a file or directory.</returns>
			boost::filesystem::path SetOutputPath(boost::filesystem::path path, const std::string extension, const std::string filename);
	};
}
#endif // __WSICS_STANDARDIZATION_CLI_H__