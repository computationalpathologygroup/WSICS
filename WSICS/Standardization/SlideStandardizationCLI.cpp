#include "SlideStandardizationCLI.h"

#include <unordered_set>

#include "../Misc/Mt_Singleton.hpp"

namespace WSICS::Standardization
{
	SlideStandardizationCLI::SlideStandardizationCLI(void)
	{
	}

	void SlideStandardizationCLI::ExecuteModuleFunctionality$(const boost::program_options::variables_map& variables)
	{
		// Configure the parameters.
		StandardizationParameters parameters(StandardizationExecution::GetStandardParameters());
		std::vector<boost::filesystem::path> files_to_process;
		std::string prefix;
		std::string postfix;
		boost::filesystem::path image_output;
		boost::filesystem::path lut_output;
		boost::filesystem::path template_input;
		boost::filesystem::path template_output;
		boost::filesystem::path debug_dir;
		bool input_is_directory;

		AcquireAndSanitizeInput_(
			variables,
			parameters,
			files_to_process,
			prefix,
			postfix,
			image_output,
			lut_output,
			template_input,
			template_output,
			debug_dir,
			input_is_directory);

		bool succesfully_created_directories = true;
		try
		{
			CreateDirectories_(image_output, lut_output, template_output, debug_dir, files_to_process, input_is_directory);
		}
		catch (...)
		{
			IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
			succesfully_created_directories = false;

			logging_instance->QueueCommandLineLogging("Unable to create directories, ending execution.", IO::Logging::SILENT);
		}

		// Sets the seed for deterministic processing.
		Misc::MT_Singleton::SetSeed(parameters.seed);

		// Attempts to utilize one of the output paths as path for the log file.
		boost::filesystem::path log_path;
		if (!image_output.empty())
		{
			log_path = image_output;
		}
		else if (!lut_output.empty())
		{
			log_path = lut_output;
		}
		else if (!template_output.empty())
		{
			log_path = template_output;
		}

		if (succesfully_created_directories && !log_path.empty())
		{
			std::string log_file;
			if (input_is_directory)
			{
				log_file = log_path.string() + "/log.txt";
			}
			else
			{
				log_path = log_path.parent_path().string() + "/log.txt";
			}

			StandardizationExecution slide_standardizer(log_file, template_input, parameters);
			for (const boost::filesystem::path& filepath : files_to_process)
			{
				std::string extension(filepath.extension().string());
				std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

				boost::filesystem::path image_output_file;
				boost::filesystem::path lut_output_file;
				boost::filesystem::path template_output_file;
				boost::filesystem::path file_debug_dir(debug_dir.string() + "/" + filepath.stem().string());

				if (input_is_directory)
				{
					image_output_file = SetOutputPath(image_output, "tif", prefix + filepath.stem().string() + postfix + "_normalized");
					lut_output_file = SetOutputPath(lut_output, "tif", prefix + filepath.stem().string() + postfix + "_lut");
					template_output_file = SetOutputPath(template_output, "csv", prefix + filepath.stem().string() + postfix);
				}
				else
				{
					image_output_file = SetOutputPath(image_output, "tif", "");
					lut_output_file = SetOutputPath(lut_output, "tif", "");
					template_output_file = SetOutputPath(template_output, "csv", "");
				}

				slide_standardizer.Normalize(filepath, image_output_file, lut_output_file, template_output_file, file_debug_dir);
			}
		}
	}

	void SlideStandardizationCLI::AddModuleOptions$(boost::program_options::options_description& options)
	{
		options.add_options()
			("input,i", boost::program_options::value<std::string>()->default_value(""), "Path to an image file or image directory.")
			("image_output", boost::program_options::value<std::string>()->default_value(""), "Path to the image output file or directory. If set, outputs the normalized WSI and potentially debug data. Should refer to a directory if the input does as well, vice versa for a file.")
			("lut_output", boost::program_options::value<std::string>()->default_value(""), "Path to the lut output file or directory. If set, outputs the LUT.Should refer to a directory if the input does as well, vice versa for a file.")
			("max_training", boost::program_options::value<uint32_t>()->default_value(20000000), "The maximum amount of pixels used for training the classifier.")
			("min_training", boost::program_options::value<uint32_t>()->default_value(200000), "The minimum amount of pixels used for training the classifier.")
			("prefix", boost::program_options::value<std::string>()->default_value(""), "The prefix to use for the output files. Only applied when the input path points towards a directory.")
			("postfix", boost::program_options::value<std::string>()->default_value(""), "The postfix to use for the output files. Only applied when the input path points towards a directory.")
			("template_input", boost::program_options::value<std::string>()->default_value(""), "If set, normalizes the source image to more closely resemble the template. The template needs to be a CSV file and can be generated by the --output_template command.")
			("template_output", boost::program_options::value<std::string>()->default_value(""), "Path to an template output file. If set, outputs the template. Should refer to a directory if the input does as well, vice versa for a file.")
			("ink,k", boost::program_options::value<bool>()->default_value(false)->implicit_value(true), "Warning: Only use if ink is present on the slide. Reduces the chance of selecting a patch containing ink.")
			("hema_percentile", boost::program_options::value<float>()->default_value(0.1f), "Defines how conservative the algorithm is with its blue pixel classification.")
			("eosin_percentile", boost::program_options::value<float>()->default_value(0.2f), "Defines how conservative the algorithm is with its red pixel classification.")
			("background_threshold", boost::program_options::value<float>()->default_value(0.9f), "Defines the threshold between tissue and background pixels.")
			("min_ellipses", boost::program_options::value<int32_t>()->default_value(false), "Allows for a custom value for the amount of ellipses on a tile.")
			("seed,s", boost::program_options::value<uint64_t>()->default_value(1000), "Defines the seed used for random processing.");
	}

	void SlideStandardizationCLI::Setup$(void)
	{
	}

	void SlideStandardizationCLI::AcquireAndSanitizeInput_(
		const boost::program_options::variables_map& variables,
		StandardizationParameters& parameters,
		std::vector<boost::filesystem::path>& files_to_process,
		std::string& prefix,
		std::string& postfix,
		boost::filesystem::path& image_output,
		boost::filesystem::path& lut_output,
		boost::filesystem::path& template_input,
		boost::filesystem::path& template_output,
		boost::filesystem::path& debug_dir,
		bool& input_is_directory)
	{
		parameters.consider_ink = variables["ink"].as<bool>();
		input_is_directory = boost::filesystem::is_directory(variables["input"].as<std::string>());

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

		parameters.max_training_size = variables["max_training"].as<uint32_t>();
		parameters.min_training_size = variables["min_training"].as<uint32_t>();

		if (parameters.max_training_size == 0)
		{
			throw std::runtime_error("The max training size requires a value greater than 0.");
		}
		else if (parameters.min_training_size > parameters.max_training_size)
		{
			uint32_t temp_min = parameters.min_training_size;
			parameters.min_training_size = parameters.max_training_size;
			parameters.max_training_size = temp_min;
		}

		parameters.seed = variables["seed"].as<uint64_t>();

		prefix = variables["prefix"].as<std::string>();
		postfix = variables["postfix"].as<std::string>();

		image_output = boost::filesystem::path(variables["image_output"].as<std::string>());
		lut_output = boost::filesystem::path(variables["lut_output"].as<std::string>());
		template_input = boost::filesystem::path(variables["template_input"].as<std::string>());
		template_output = boost::filesystem::path(variables["template_output"].as<std::string>());

		if (!image_output.empty())
		{
			if (image_output.has_extension())
			{
				image_output = image_output.parent_path().append("/" + image_output.stem().string());
			}
		}

		if (!lut_output.empty() && lut_output.has_extension())
		{
			lut_output = lut_output.parent_path().append("/" + lut_output.stem().string());
		}

		if (!template_input.empty() && !boost::filesystem::is_regular_file(template_input))
		{
			throw std::runtime_error("The template input path points towards an invalid file.");
		}

		if (!template_output.empty() && template_output.has_extension())
		{
			template_output = template_output.parent_path().append("/" + template_output.stem().string());
		}

		if (IO::Logging::LogHandler::GetInstance()->GetOutputLevel() == IO::Logging::DEBUG)
		{
			boost::filesystem::path debug_potential;
			if (!image_output.empty())
			{
				debug_potential = image_output;
			}
			else if (!lut_output.empty())
			{
				debug_potential = lut_output;
			}

			if (input_is_directory && !debug_potential.empty())
			{
				debug_dir = debug_potential.string() + "/debug";
			}
			else
			{
				debug_dir = debug_potential.parent_path().string() + "/debug";
			}
		}

		parameters.hema_percentile = variables["hema_percentile"].as<float>();
		parameters.eosin_percentile = variables["eosin_percentile"].as<float>();
		parameters.background_threshold = variables["background_threshold"].as<float>();
		parameters.minimum_ellipses = variables["min_ellipses"].as<int32_t>();

		if (parameters.hema_percentile > 1.0f)
		{
			parameters.hema_percentile = 1.0f;
		}
		if (parameters.eosin_percentile > 1.0f)
		{
			parameters.eosin_percentile = 1.0f;
		}
		if (parameters.background_threshold > 1.0f)
		{
			parameters.background_threshold = 1.0f;
		}


	}

	void SlideStandardizationCLI::CreateDirectories_(
		const boost::filesystem::path& image_output,
		const boost::filesystem::path& lut_output,
		const boost::filesystem::path& template_output,
		const boost::filesystem::path& debug_directory,
		const std::vector<boost::filesystem::path>& files,
		const bool input_is_directory
	)
	{
		IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

		if (!image_output.empty())
		{
			if (input_is_directory)
			{
				boost::filesystem::create_directories(image_output);
				logging_instance->QueueCommandLineLogging("Created: " + image_output.string(), IO::Logging::NORMAL);
			}
			else
			{
				boost::filesystem::create_directories(image_output.parent_path());
				logging_instance->QueueCommandLineLogging("Created: " + image_output.parent_path().string(), IO::Logging::NORMAL);
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

	boost::filesystem::path SlideStandardizationCLI::SetOutputPath(boost::filesystem::path path, const std::string extension, const std::string filename)
	{
		if (!path.empty())
		{
			path = path.string() + filename + "." + extension;
		}
		return path;
	}
}