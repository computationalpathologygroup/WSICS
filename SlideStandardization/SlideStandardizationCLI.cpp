#include "SlideStandardizationCLI.h"

#include <unordered_set>

#include "Standardization.hpp"

SlideStandardizationCLI::SlideStandardizationCLI(void)
{
}

void SlideStandardizationCLI::ExecuteModuleFunctionality$(const boost::program_options::variables_map& variables)
{
	std::vector<boost::filesystem::path> files_to_process;
	uint32_t max_training_size;
	uint32_t min_training_size;
	boost::filesystem::path output_dir;
	boost::filesystem::path template_dir;
	boost::filesystem::path debug_dir;
	bool contains_ink;
	bool write_template;
	bool write_wsi;

	AcquireAndSanitizeInput_(variables, files_to_process, max_training_size, min_training_size, output_dir, template_dir, debug_dir, contains_ink, write_template, write_wsi);

	boost::filesystem::create_directory(debug_dir);

	Standardization slide_standardizer("log.txt");
	for (const boost::filesystem::path& filepath : files_to_process)
	{
		std::string extension = filepath.extension().string();
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
		bool is_tiff = extension == ".tif";

		//boost::filesystem::path output_filepath(output_dir.append(filepath.stem().append("_normalized.tif")));




		uint32_t tilesize = 512;
		//slide_standardizer.CreateNormalizationLUT(filepath, template_path, output_path, output_path + "/debug/", max_training_size, min_training_size, tilesize, is_tiff, write_template, contains_ink);

		if (write_wsi)
		{

		}
	}
}

void SlideStandardizationCLI::AddModuleOptions$(boost::program_options::options_description& options)
{
	options.add_options()
		("input,i",			boost::program_options::value<std::string>(),										"Path to an image file or image directory.")
		("template,t",		boost::program_options::value<bool>()->default_value(false)->implicit_value(true),	"Only outputs the template.")
		("output,o",		boost::program_options::value<std::string>(),										"Path to the output directory.")
		("max_training",	boost::program_options::value<uint32_t>()->default_value(20000000),					"The maximum amount of pixels used for training the classifier.")
		("min_training",	boost::program_options::value<uint32_t>()->default_value(200000),					"The minimum amount of pixels used for training the classifier.")
		("write_wsi,w",		boost::program_options::value<bool>()->default_value(false)->implicit_value(true),	"If set, the normalized image will be written towards the filesystem.")
		("template_dir",	boost::program_options::value<std::string>(),										"The directory where templates are stored and written to.")
		("ink,k",			boost::program_options::value<bool>()->default_value(false)->implicit_value(true),	"Warning: Only use if ink is present on the slide. Reduces the chance of selecting a patch containing ink.");
}

void SlideStandardizationCLI::Setup$(void)
{
}

void SlideStandardizationCLI::AcquireAndSanitizeInput_(const boost::program_options::variables_map& variables,
	std::vector<boost::filesystem::path>& files_to_process,
	uint32_t& max_training_size,
	uint32_t& min_training_size,
	boost::filesystem::path& output_dir,
	boost::filesystem::path& template_dir,
	boost::filesystem::path& debug_dir,
	bool& contains_ink,
	bool& write_template,
	bool& write_wsi)
{
	try
	{
		std::vector<boost::filesystem::path> files_to_process = GatherImageFilenames_(boost::filesystem::path(variables["input"].as<std::string>()));

		if (files_to_process.empty())
		{
			throw std::runtime_error("No files to process"); // Redundant, but triggers the catch block without any additional checks.
		}
	}
	catch (...)
	{
		throw std::runtime_error("Unable access or acquire any valid files from the input path.");
	}

	max_training_size = variables["max_training"].as<uint32_t>();
	min_training_size = variables["min_training"].as<uint32_t>();

	if (max_training_size == 0)
	{
		throw std::runtime_error("The max training size requires a positive value.");
	}
	else if (min_training_size > max_training_size)
	{
		uint32_t temp_min = min_training_size;
		min_training_size = max_training_size;
		max_training_size = temp_min;
	}

	output_dir = boost::filesystem::path(variables["output"].as<std::string>());
	template_dir = boost::filesystem::path(variables["template_dir"].as<std::string>());

	if (!output_dir.empty() && !boost::filesystem::is_directory(output_dir))
	{
		throw std::runtime_error("The output path points towards an invalid directory.");
	}
	if (!template_dir.empty() && !boost::filesystem::is_directory(template_dir))
	{
		throw std::runtime_error("The template path points towards an invalid directory.");
	}

	debug_dir = output_dir.append("debug");

	contains_ink = variables["ink"].as<bool>();
	write_template = variables["template"].as<bool>();
	write_wsi = variables["write_wsi"].as<bool>();
}

std::vector<boost::filesystem::path> SlideStandardizationCLI::GatherImageFilenames_(const boost::filesystem::path input_path)
{
	// TODO: These should be pulled from the image loading DLL/SO.
	std::unordered_set<std::string> files_to_accept;
	files_to_accept.insert(".ndpi");
	files_to_accept.insert(".tif");
	files_to_accept.insert(".mrxs");
	files_to_accept.insert(".svs");
	files_to_accept.insert(".vsi");

	std::vector<boost::filesystem::path> files;
	if (boost::filesystem::is_directory(input_path))
	{
		boost::filesystem::directory_iterator begin(input_path), end;
		for (; begin != end; ++begin)
		{
			std::string extension = begin->path().extension().string();
			std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

			if (files_to_accept.find(extension) != files_to_accept.end())
			{
				files.push_back(begin->path());
			}
		}
	}
	else
	{
		std::string extension = input_path.extension().string();
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

		if (files_to_accept.find(extension) != files_to_accept.end())
		{
			files.push_back(input_path);
		}
	}

	return files;
}