#include "SlideStandardizationCLI.h"

#include <unordered_set>

#include "Standardization.h"

SlideStandardizationCLI::SlideStandardizationCLI(void)
{
}

void SlideStandardizationCLI::ExecuteModuleFunctionality$(const boost::program_options::variables_map& variables)
{
	std::vector<boost::filesystem::path> files_to_process;
	uint32_t max_training_size;
	uint32_t min_training_size;
	boost::filesystem::path output_path;
	boost::filesystem::path template_input;
	boost::filesystem::path template_output;
	boost::filesystem::path debug_dir;
	bool contains_ink;

	AcquireAndSanitizeInput_(variables, files_to_process, max_training_size, min_training_size, output_path, template_input, template_output, debug_dir, contains_ink);

	bool succesfully_created_directories = true;
	try
	{
		CreateDirectories_(output_path, template_output, debug_dir);
	}
	catch (...)
	{
		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
		succesfully_created_directories = false;

		logging_instance->QueueCommandLineLogging("Unable to create directories, ending execution.", IO::Logging::SILENT);
	}
	
	if (succesfully_created_directories && (!output_path.empty() || !template_output.empty()))
	{
		Standardization slide_standardizer(output_path.string() + "log.txt", template_input, debug_dir, min_training_size, max_training_size);
		for (const boost::filesystem::path& filepath : files_to_process)
		{
			std::string extension(filepath.extension().string());
			std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

			bool is_tiff = extension == ".tif";

			boost::filesystem::path output_file;
			boost::filesystem::path template_output_file;

			if (files_to_process.size() > 1)
			{
				output_file				= output_path.string() + ".tif";
				template_output_file	= template_output.string() + ".csv";
			}
			else
			{
				output_file				= output_path.string() + filepath.stem().string() + "_normalized.tif";
				template_output_file	= template_output.string() +filepath.stem().string() + ".csv";
			}

			slide_standardizer.Normalize(filepath, output_file, template_output_file, is_tiff, contains_ink);
		}
	}
}

void SlideStandardizationCLI::AddModuleOptions$(boost::program_options::options_description& options)
{
	options.add_options()
		("input,i",			boost::program_options::value<std::string>()->default_value(""),					"Path to an image file or image directory.")
		("output,o",		boost::program_options::value<std::string>()->default_value(""),					"Path to the output file. If set, outputs the LUT and normalized WSI. Serves as a prefix if the input points towards a directory.")
		("max_training",	boost::program_options::value<uint32_t>()->default_value(20000000),					"The maximum amount of pixels used for training the classifier.")
		("min_training",	boost::program_options::value<uint32_t>()->default_value(200000),					"The minimum amount of pixels used for training the classifier.")
		("template_input",	boost::program_options::value<std::string>()->default_value(""),					"If set, applies an existing template for the normalization.")
		("template_output", boost::program_options::value<std::string>()->default_value(""),					"Path to an template output file. If set, outputs the template.Serves as a prefix if the input points towards a directory.")
		("ink,k",			boost::program_options::value<bool>()->default_value(false)->implicit_value(true),	"Warning: Only use if ink is present on the slide. Reduces the chance of selecting a patch containing ink.");
}

void SlideStandardizationCLI::Setup$(void)
{
}

void SlideStandardizationCLI::AcquireAndSanitizeInput_(
	const boost::program_options::variables_map& variables,
	std::vector<boost::filesystem::path>& files_to_process,
	uint32_t& max_training_size,
	uint32_t& min_training_size,
	boost::filesystem::path& output_path,
	boost::filesystem::path& template_input,
	boost::filesystem::path& template_output,
	boost::filesystem::path& debug_dir,
	bool& contains_ink)
{
	try
	{
		files_to_process = GatherImageFilenames_(boost::filesystem::path(variables["input"].as<std::string>()));

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
		throw std::runtime_error("The max training size requires a value greater than 0.");
	}
	else if (min_training_size > max_training_size)
	{
		uint32_t temp_min = min_training_size;
		min_training_size = max_training_size;
		max_training_size = temp_min;
	}

	output_path		= boost::filesystem::path(variables["output"].as<std::string>());
	template_input	= boost::filesystem::path(variables["template_input"].as<std::string>());
	template_output = boost::filesystem::path(variables["template_output"].as<std::string>());

	if (!output_path.empty())
	{
		if (output_path.has_extension() && files_to_process.size() > 1)
		{
			output_path = output_path.parent_path().append(output_path.stem().string());
		}
		else
		{
			output_path.append("/");
		}

		if (IO::Logging::LogHandler::GetInstance()->GetOutputLevel() == IO::Logging::DEBUG)
		{
			if (output_path.has_filename())
			{
				debug_dir = output_path.parent_path().append("debug");
			}
			else
			{
				debug_dir = output_path.append("debug");
			}
		}
	}
	if (!template_input.empty() && !boost::filesystem::is_regular_file(template_input))
	{
		throw std::runtime_error("The template input path points towards an invalid file.");
	}

	if (!template_output.empty())
	{
		if (template_output.has_extension() && files_to_process.size() > 1)
		{
			template_output = template_output.parent_path().append(template_output.stem().string());
		}
		else
		{
			template_output.append("/");
		}
	}

	contains_ink	= variables["ink"].as<bool>();
}

void SlideStandardizationCLI::CreateDirectories_(const boost::filesystem::path& output_path, const boost::filesystem::path& template_output, const boost::filesystem::path& debug_directory)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	if (!output_path.empty())
	{
		if (output_path.has_extension())
		{
			boost::filesystem::create_directories(output_path.parent_path());
			logging_instance->QueueCommandLineLogging("Created: " + output_path.parent_path().string(), IO::Logging::NORMAL);
		}
		else
		{
			boost::filesystem::create_directories(output_path);
			logging_instance->QueueCommandLineLogging("Created: " + output_path.string(), IO::Logging::NORMAL);
		}
	}

	if (!template_output.empty())
	{
		if (template_output.has_extension())
		{
			boost::filesystem::create_directories(template_output.parent_path());
			logging_instance->QueueCommandLineLogging("Created: " + template_output.parent_path().string(), IO::Logging::NORMAL);
		}
		else
		{
			boost::filesystem::create_directories(template_output);
			logging_instance->QueueCommandLineLogging("Created: " + template_output.string(), IO::Logging::NORMAL);
		}
	}

	if (!debug_directory.empty())
	{
		boost::filesystem::create_directory(debug_directory);
		logging_instance->QueueCommandLineLogging("Created: " + debug_directory.string(), IO::Logging::NORMAL);
	}
}

std::vector<boost::filesystem::path> SlideStandardizationCLI::GatherImageFilenames_(const boost::filesystem::path input_path)
{
	// TODO: These should be pulled from the image loading DLL/SO.
	std::unordered_set<std::string> filetypes_to_accept;
	filetypes_to_accept.insert(".ndpi");
	filetypes_to_accept.insert(".tif");
	filetypes_to_accept.insert(".mrxs");
	filetypes_to_accept.insert(".svs");
	filetypes_to_accept.insert(".vsi");

	std::vector<boost::filesystem::path> files;
	if (boost::filesystem::is_directory(input_path))
	{
		boost::filesystem::directory_iterator begin(input_path), end;
		for (; begin != end; ++begin)
		{
			std::string extension = begin->path().extension().string();
			std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

			if (filetypes_to_accept.find(extension) != filetypes_to_accept.end())
			{
				files.push_back(begin->path());
			}
		}
	}
	else
	{
		std::string extension = input_path.extension().string();
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

		if (filetypes_to_accept.find(extension) != filetypes_to_accept.end())
		{
			files.push_back(input_path);
		}
	}

	return files;
}