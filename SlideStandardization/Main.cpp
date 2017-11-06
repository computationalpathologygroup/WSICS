#include "Standardization.hpp"
#include <iostream>

#include "IO/Logging/LogHandler.h"

#include <boost/program_options.hpp>

using namespace boost::filesystem;
using namespace std;
namespace po = boost::program_options;

std::vector<std::string> GetFilenames(path directory)
{
	std::vector<std::string> FileNames;
	directory_iterator iter(directory), end;
	for (; iter != end; ++iter){
		if (iter->path().extension() == ".tif")
			FileNames.emplace_back(iter->path().filename().string());
	}

	return FileNames;
}

int main( int argc, char * argv[])
{
	po::options_description description("Standardization options");
	description.add_options()
		("input,i", po::value<std::string>(), "Input image file or image DIR")
		("template,t", po::value<bool>()->default_value(false)->implicit_value(true), "Define Template")
		("output,o", po::value<std::string>(), "Output DIR")
		("num1,n", po::value<int>()->default_value(20000000), "Number of samples for WSI")
		("num2,m", po::value<int>()->default_value(200000), "Min number of samples for a patch")
		("writeWSI,w", po::value<bool>()->default_value(false)->implicit_value(true), "min number of samples for a patch")
		("templateDIR,d", po::value<std::string>(), "Output DIR")
		("ink,k", po::value<bool>()->default_value(false)->implicit_value(true), "Suppress ink")
		("help,h", po::value<bool>()->default_value(false)->implicit_value(true), "Help instructions");

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
	po::notify(vm);

	if (vm["help"].as<bool>())
	{
		cout << "(Optional) -help or -h          Shows this message." << endl;
		cout << "(Required) -input or -i         Input image filename (can be .tiff, .mrxs, .ndpi, etc.) or directory containing all images." << endl;
		cout << "(Optional) -template or -t      Define template parameters (type -t to define template, skip this parameter otherwise.)" << endl;
		cout << "(Optional) -output or -o        Directory to store the standardized image. By default this is the same directory as the input image." << endl;
		cout << "(Optional) -num1 or -n          The number of samples to pick up from WSI for normalization. Default is 20,000,000." << endl;
		cout << "(Optional) -num2 or -m          Minimum number of samples needed to pick from a single patch. Default is 200,000." << endl;
		cout << "(Optional) -writeWSI or -w      write normalized WSI to disk?" << endl;
		cout << "(Optional) -templateDIR or -d   The directory for storing or loading the template csv file. By default this directory is the dir of the excecutable." << endl;
		cout << "(Optional) -ink or -k           Type -k if there is ink on the slide. This will reduce the chance of picking a patch containing ink as example." << endl;
		return 0;
	}

	string filename_input;
	if (vm.count("input"))
		filename_input = vm["input"].as<string>();
	else
	{
		cout << "It is required to provide an input path to image." << endl;
		cout << "To get help type: -h or -help" << endl;
		return 0;
	}

	bool DefineTemplate = vm["template"].as<bool>();
	int trainingSizeInput = vm["num1"].as<int>();
	int MinTrainSize = vm["num2"].as<int>();
	bool write_std_WSI = vm["writeWSI"].as<bool>();
	bool ink = vm["ink"].as<bool>();



	// TODO: Replace
	std::string DebugDIR = "debug";

	bool IsTiff = 1;
	// extract the filename by removing the path from it!

	path directory(filename_input);
	std::vector<std::string> FileNames;
	
	if (!DefineTemplate){
		if (is_directory(directory))
		{
			directory_iterator iter(directory), end;
			for (; iter != end; ++iter){
				if (iter->path().extension() == ".ndpi" || iter->path().extension() == ".tif" || iter->path().extension() == ".mrxs" || iter->path().extension() == ".svs" || iter->path().extension() == ".vsi")
					FileNames.push_back(iter->path().filename().string());
			}
		}
		else
		{
			FileNames.push_back(filename_input);
		}
	}
	else{
		FileNames.push_back(filename_input);
	}

	
	for (int k = 0; k < FileNames.size(); ++k)
	{
		// Read ground truth database
		std::string Current_Filename;


		if (!DefineTemplate && is_directory(directory)){
			std::string dash = "\\";
			Current_Filename = filename_input + dash + FileNames[k];
		}
		else{
			Current_Filename = filename_input;
		}

		if (Current_Filename.substr(Current_Filename.find_last_of(".") + 1) == "tif") {
			IsTiff = 1;
		}

		string FullFN, FileName;
		size_t sep = Current_Filename.find_last_of("\\/");
		FullFN = Current_Filename.substr(sep + 1, Current_Filename.size() - sep - 1);

		size_t dot = FullFN.find_last_of(".");
		FileName = FullFN.substr(0, dot);
		string SaveDir = Current_Filename.substr(0, Current_Filename.rfind(".")) + "_Normalized.tif";
		string CurrentDIR = Current_Filename.substr(0, Current_Filename.rfind("\\"));
		
		string log_dir = DebugDIR + "\\" + FileName + ".txt";
		std::ofstream log_file(log_dir, std::ios_base::out | std::ios_base::app);
		
		string log_text = "Normalizing image: " + FileName;
		log_file << log_text << endl;
		cout << "Standardizing slide: " + FileName + " - Image number " << k + 1 << "/" << FileNames.size() <<endl;
		if (vm.count("output"))
		{
			string output_dir = vm["output"].as<string>();
			path Sdirectory(output_dir);
			SaveDir = output_dir;
			sep = SaveDir.find(".tif");
			if (sep == std::string::npos)
				SaveDir = SaveDir + "\\" + FileName + "_Normalized.tif";
		}
		
		string TemplateDIR;
		if (vm.count("templateDIR"))
			TemplateDIR = vm["templateDIR"].as<string>();
		else
			TemplateDIR = DebugDIR + "\\Template.csv";

		log_text = "Saving image DIR: " + SaveDir;
		log_file << log_text << endl;
		log_text = "Saving/loading template DIR: " + TemplateDIR;
		log_file << log_text << endl;
		log_text = "Total number of samples for WSI: " + std::to_string(trainingSizeInput);
		log_file << log_text << endl;
		log_text = "Min number of samples per patch: " + std::to_string(MinTrainSize);
		log_file << log_text << endl;
		log_file << "Defining template parameters: " + to_string(DefineTemplate) << endl;
		log_file << "Writing WSI to disk: " + to_string(write_std_WSI) << endl;
		log_file << "Considering ink: " + to_string(ink) << endl;


		IO::Logging::LogHandler log_handler(IO::Logging::NORMAL);
		IO::Logging::LogHandler::Register(&log_handler);
		log_handler.Initialize();

		std::string parameter_filepath("");

		std::string test("D:/WSIs/test.ndpi");
		

		int tilesize = 512;	
		Standardization standardization("logs");
		standardization.CreateNormalizationLUT(test, parameter_filepath, SaveDir, DebugDIR, trainingSizeInput, MinTrainSize, tilesize, IsTiff, 0, ink);
		//standardization.CreateStandardizationLUT(Current_Filename.c_str(), TemplateDIR, SaveDir, trainingSizeInput, tilesize, IsTiff, MinTrainSize, DefineTemplate, log_dir, ink);
		
		/*if (!DefineTemplate && write_std_WSI == 1 && standardizedImage.ImageIsMultiresolution)
		{
			cout << "Writing the standardized WSI in progress..." << endl;
			log_file << "Writing the standardized WSI in progress..." << endl;
			standardizedImage.writeNormalizedWSI(Current_Filename.c_str(), standardizedImage.channelsTileN, tilesize, IsTiff, SaveDir);
			cout << "Finished writing the image." << endl;
			log_file << "Finished writing the image." << endl;
		}*/
	}
	//system("pause");
	return 0;
}
