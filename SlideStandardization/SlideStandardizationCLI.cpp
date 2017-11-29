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
	std::string prefix;
	std::string postfix;
	boost::filesystem::path output_path;
	boost::filesystem::path template_input;
	boost::filesystem::path template_output;
	boost::filesystem::path debug_dir;
	bool contains_ink;
	bool input_is_directory;

	AcquireAndSanitizeInput_(
		variables,
		files_to_process,
		max_training_size,
		min_training_size,
		prefix,
		postfix,
		output_path,
		template_input,
		template_output,
		debug_dir,
		contains_ink,
		input_is_directory);

	bool succesfully_created_directories = true;
	try
	{
		CreateDirectories_(output_path, template_output, debug_dir, files_to_process, input_is_directory);
	}
	catch (...)
	{
		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
		succesfully_created_directories = false;

		logging_instance->QueueCommandLineLogging("Unable to create directories, ending execution.", IO::Logging::SILENT);
	}
	
	if (succesfully_created_directories && (!output_path.empty() || !template_output.empty()))
	{
		Standardization slide_standardizer(output_path.string() + "log.txt", template_input, min_training_size, max_training_size);
		for (const boost::filesystem::path& filepath : files_to_process)
		{
			std::string extension(filepath.extension().string());
			std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

			bool is_tiff = extension == ".tif";

			boost::filesystem::path output_file;
			boost::filesystem::path template_output_file;
			boost::filesystem::path file_debug_dir(debug_dir.string() + "/" + filepath.stem().string());

			if (input_is_directory)
			{
				output_file				= output_path.string()		+ prefix + filepath.stem().string() + postfix + "_normalized.tif";
				template_output_file	= template_output.string()	+ prefix + filepath.stem().string() + postfix + ".csv";
			}
			else
			{
				output_file				= output_path.string() + ".tif";
				template_output_file	= template_output.string() + ".csv";
			}

			slide_standardizer.Normalize(filepath, output_file, template_output_file, file_debug_dir, is_tiff, contains_ink);
		}
	}
}

void SlideStandardizationCLI::AddModuleOptions$(boost::program_options::options_description& options)
{
	options.add_options()
		("input,i",			boost::program_options::value<std::string>()->default_value(""),					"Path to an image file or image directory.")
		("output,o",		boost::program_options::value<std::string>()->default_value(""),					"Path to the output file. If set, outputs the LUT and normalized WSI. Considered as filepath if input points towards a file, otherwise considered as output directory.")
		("max_training",	boost::program_options::value<uint32_t>()->default_value(20000000),					"The maximum amount of pixels used for training the classifier.")
		("min_training",	boost::program_options::value<uint32_t>()->default_value(200000),					"The minimum amount of pixels used for training the classifier.")
		("prefix",			boost::program_options::value<std::string>()->default_value(""),					"The prefix to use for the output files. Only applied when the input path points towards a directory.")
		("postfix",		boost::program_options::value<std::string>()->default_value(""),						"The postfix to use for the output files. Only applied when the input path points towards a directory.")
		("template_input",	boost::program_options::value<std::string>()->default_value(""),					"If set, applies an existing template for the normalization.")
		("template_output", boost::program_options::value<std::string>()->default_value(""),					"Path to an template output file. If set, outputs the template.  Considered as filepath if input points towards a file, otherwise considered as output directory.")
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
	std::string& prefix,
	std::string& postfix,
	boost::filesystem::path& output_path,
	boost::filesystem::path& template_input,
	boost::filesystem::path& template_output,
	boost::filesystem::path& debug_dir,
	bool& contains_ink,
	bool& input_is_directory)
{
	contains_ink		= variables["ink"].as<bool>();
	input_is_directory	= boost::filesystem::is_directory(variables["input"].as<std::string>());

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

	prefix	= variables["prefix"].as<std::string>();
	postfix = variables["postfix"].as<std::string>();

	output_path		= boost::filesystem::path(variables["output"].as<std::string>());
	template_input	= boost::filesystem::path(variables["template_input"].as<std::string>());
	template_output = boost::filesystem::path(variables["template_output"].as<std::string>());

	if (!output_path.empty())
	{
		if (output_path.has_extension())
		{
			output_path = output_path.parent_path().append("/" + output_path.stem().string());
		}

		if (IO::Logging::LogHandler::GetInstance()->GetOutputLevel() == IO::Logging::DEBUG)
		{
			if (input_is_directory)
			{
				debug_dir = output_path.string() + "/debug";
			}
			else
			{
				debug_dir = output_path.parent_path().string() + "/debug";
			}
		}
	}
	if (!template_input.empty() && !boost::filesystem::is_regular_file(template_input))
	{
		throw std::runtime_error("The template input path points towards an invalid file.");
	}

	if (!template_output.empty() && template_output.has_extension())
	{
		template_output = template_output.parent_path().append("/" + template_output.stem().string());
	}
}

void SlideStandardizationCLI::CreateDirectories_(
	const boost::filesystem::path& output_path,
	const boost::filesystem::path& template_output,
	const boost::filesystem::path& debug_directory,
	const std::vector<boost::filesystem::path>& files,
	const bool input_is_directory
)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	if (!output_path.empty())
	{
		if (input_is_directory)
		{
			boost::filesystem::create_directories(output_path);
			logging_instance->QueueCommandLineLogging("Created: " + output_path.string(), IO::Logging::NORMAL);
		}
		else
		{
			boost::filesystem::create_directories(output_path.parent_path());
			logging_instance->QueueCommandLineLogging("Created: " + output_path.parent_path().string(), IO::Logging::NORMAL);
		}
	}

	if (!template_output.empty())
	{
		if (input_is_directory)
		{
			boost::filesystem::create_directories(template_output);
			logging_instance->QueueCommandLineLogging("Created: " + template_output.string(), IO::Logging::NORMAL);
		}
		else
		{
			boost::filesystem::create_directories(template_output.parent_path());
			logging_instance->QueueCommandLineLogging("Created: " + template_output.parent_path().string(), IO::Logging::NORMAL);
		}
	}

	if (!debug_directory.empty())
	{
		boost::filesystem::create_directory(debug_directory);
		logging_instance->QueueCommandLineLogging("Created: " + debug_directory.string(), IO::Logging::NORMAL);

		for (const boost::filesystem::path& filepath : files)
		{
			std::string debug_base(debug_directory.string() + "/" + filepath.stem().string());
			boost::filesystem::create_directories(debug_base);
			logging_instance->QueueCommandLineLogging("Created: " + debug_base, IO::Logging::NORMAL);

			boost::filesystem::create_directory(debug_base + "/classification_result");
			logging_instance->QueueCommandLineLogging("Created: " + debug_base + "/classification_result", IO::Logging::NORMAL);

			boost::filesystem::create_directory(debug_base + "/normalized_examples");
			logging_instance->QueueCommandLineLogging("Created: " + debug_base + "/normalized_examples", IO::Logging::NORMAL);

			boost::filesystem::create_directory(debug_base + "/raw_tiles");
			logging_instance->QueueCommandLineLogging("Created: " + debug_base + "/raw_tiles", IO::Logging::NORMAL);
		}
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