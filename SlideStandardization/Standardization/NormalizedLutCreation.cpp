#include "NormalizedLutCreation.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include "../Misc/LevelReading.h"
#include "../HSD/Transformations.h"
#include "../IO/Logging/LogHandler.h"

cv::Mat matread(const std::string& filename)
{
	std::ifstream fs(filename, std::fstream::binary);

	// Header
	int rows, cols, type, channels;
	fs.read((char*)&rows, sizeof(int));         // rows
	fs.read((char*)&cols, sizeof(int));         // cols
	fs.read((char*)&type, sizeof(int));         // type
	fs.read((char*)&channels, sizeof(int));     // channels

												// Data
	cv::Mat mat(rows, cols, type);
	fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

	return mat;
}

cv::Mat NormalizedLutCreation::Create(
	const bool generate_lut,
	const boost::filesystem::path& template_file,
	const boost::filesystem::path& template_output,
	const HSD::HSD_Model& lut_hsd,
	const TrainingSampleInformation& training_samples,
	const uint32_t max_training_size,
	const size_t log_file_id)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	//===========================================================================
	//	Transforming Cx and Cy distributions - Initialization 1
	//  Extracting all the parameters needed for applying the transformation
	//===========================================================================
	logging_instance->QueueFileLogging("Defining variables for transformation...", log_file_id, IO::Logging::NORMAL);

	ClassAnnotatedCxCy train_data(TransformCxCyDensity::ClassCxCyGenerator(training_samples.class_data, training_samples.training_data_cx_cy));

	// Rotates the cx_cy matrice per class and stores the parameters used.
	cv::Mat cx_cy_hema_rotated;
	cv::Mat cx_cy_eosin_rotated;
	cv::Mat cx_cy_background_Rotated;

	MatrixRotationParameters hema_rotation_info(TransformCxCyDensity::RotateCxCy(train_data.cx_cy_merged, cx_cy_hema_rotated, train_data.hema_cx_cy));
	MatrixRotationParameters eosin_rotation_info(TransformCxCyDensity::RotateCxCy(train_data.cx_cy_merged, cx_cy_eosin_rotated, train_data.eosin_cx_cy));
	MatrixRotationParameters background_rotation_info(TransformCxCyDensity::RotateCxCy(train_data.cx_cy_merged, cx_cy_background_Rotated, train_data.background_cx_cy));

	ClassPixelIndices class_pixel_indices(TransformCxCyDensity::GetClassIndices(training_samples.class_data));

	// Calculates the scale parameters per class.
	cv::Mat hema_scale_parameters(TransformCxCyDensity::CalculateScaleParameters(class_pixel_indices.hema_indices, cx_cy_hema_rotated));
	cv::Mat eosin_scale_parameters(TransformCxCyDensity::CalculateScaleParameters(class_pixel_indices.eosin_indices, cx_cy_eosin_rotated));
	cv::Mat background_scale_parameters(TransformCxCyDensity::CalculateScaleParameters(class_pixel_indices.background_indices, cx_cy_background_Rotated));

	ClassDensityRanges class_density_ranges(TransformCxCyDensity::GetDensityRanges(training_samples.class_data, training_samples.training_data_density, class_pixel_indices));

	logging_instance->QueueFileLogging("Finished computing tranformation parameters for the current image", log_file_id, IO::Logging::NORMAL);

	//===========================================================================
	//	Prepares the weight generation.
	//===========================================================================
	// Downsample the number of samples for NB classifier
	uint32_t downsample = 20;
	if (max_training_size > 10000000 && max_training_size < 20000000)
	{
		downsample = 30;
	}
	else if (max_training_size >= 20000000 && max_training_size < 30000000)
	{
		downsample = 40;
	}
	else if (max_training_size >= 30000000)
	{
		downsample = 50;
	}

	logging_instance->QueueFileLogging("Down sampling the data for constructing NB classifier", log_file_id, IO::Logging::NORMAL);

	TrainingSampleInformation sample_info_downsampled(DownsampleforNbClassifier(training_samples, downsample, max_training_size));

	logging_instance->QueueFileLogging("Generating weights with NB classifier", log_file_id, IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Generating the weights, Setting dataset of size " + std::to_string(sample_info_downsampled.class_data.rows * sample_info_downsampled.class_data.cols), IO::Logging::NORMAL);

	ML::NaiveBayesClassifier classifier(CxCyWeights::CreateNaiveBayesClassifier(sample_info_downsampled.training_data_cx_cy.col(0), sample_info_downsampled.training_data_cx_cy.col(1), sample_info_downsampled.training_data_density, sample_info_downsampled.class_data));
	//auto classifier = CxCyWeights::CreateNaiveBayesClassifier2(sample_info_downsampled.training_data_cx_cy.col(0), sample_info_downsampled.training_data_cx_cy.col(1), sample_info_downsampled.training_data_density, sample_info_downsampled.class_data);
	
	logging_instance->QueueCommandLineLogging("Training Naive Bayes Classifier fininshed...", IO::Logging::NORMAL);

	//===========================================================================
	//	Generates the weights
	//===========================================================================
	logging_instance->QueueCommandLineLogging("Generating posteriors (This will take some time...)", IO::Logging::NORMAL);
	CxCyWeights::Weights weights(CxCyWeights::GenerateWeights(lut_hsd.c_x, lut_hsd.c_y, lut_hsd.density, classifier));
	logging_instance->QueueCommandLineLogging("All weights created...", IO::Logging::NORMAL);
	logging_instance->QueueFileLogging("Weights generated", log_file_id, IO::Logging::NORMAL);

	//===========================================================================
	//	Defining Template Parameters
	//===========================================================================
	TransformationParameters calculated_transform_parameters{ hema_rotation_info, eosin_rotation_info, background_rotation_info, hema_scale_parameters, eosin_scale_parameters, class_density_ranges };
	TransformationParameters lut_transform_parameters(HandleParameterization(calculated_transform_parameters, template_file, template_output, log_file_id)); // Copies the calculated_transform_parameters or reads a new set from the offered filepath.

	if (!generate_lut)
	{
		return cv::Mat();
	}

	//===========================================================================
	//	Transforming Cx and Cy distributions - Initialization
	//===========================================================================
	logging_instance->QueueCommandLineLogging("Transformation started...", IO::Logging::NORMAL);
	logging_instance->QueueFileLogging("Transformation started...", log_file_id, IO::Logging::NORMAL);

	cv::Mat lut_cx_cy;
	cv::hconcat(std::vector<cv::Mat>{ lut_hsd.c_x, lut_hsd.c_y }, lut_cx_cy);
	std::vector<cv::Mat> lut_transformation_results(InitializeTransformation(training_samples.training_data_cx_cy, lut_cx_cy, cx_cy_hema_rotated, cx_cy_eosin_rotated, calculated_transform_parameters, lut_transform_parameters, class_pixel_indices));

	//===========================================================================
	//	Generating the weights for each class
	//===========================================================================
	logging_instance->QueueFileLogging("Applying weights...", log_file_id, IO::Logging::NORMAL);

	cv::Mat cx_cy_normalized(CxCyWeights::ApplyWeights(lut_transformation_results[0], lut_transformation_results[1], lut_transformation_results[2], weights));

	//===========================================================================
	//	Density scaling
	//===========================================================================
	logging_instance->QueueFileLogging("Density transformation...", log_file_id, IO::Logging::NORMAL);
	cv::Mat density_scaling(TransformCxCyDensity::DensityNormalizationThreeScales(calculated_transform_parameters.class_density_ranges, lut_transform_parameters.class_density_ranges, lut_hsd.density, weights));

	//===========================================================================
	//	HSD reverse
	//===========================================================================
	logging_instance->QueueFileLogging("HSD reverse...", log_file_id, IO::Logging::NORMAL);
	cv::Mat normalized_image_rgb;
	HSD::CxCyToRGB(cx_cy_normalized, normalized_image_rgb, density_scaling);
	return normalized_image_rgb;
}

TrainingSampleInformation NormalizedLutCreation::DownsampleforNbClassifier(const TrainingSampleInformation& training_samples, const uint32_t downsample, const uint32_t max_training_size)
{
	TrainingSampleInformation sample_info_downsampled
	{
		cv::Mat::zeros(max_training_size / downsample, 2, CV_32FC1),
		cv::Mat::zeros(max_training_size / downsample, 1, CV_32FC1),
		cv::Mat::zeros(max_training_size / downsample, 1, CV_32FC1),
	};

	for (size_t pixel = 0; pixel < training_samples.class_data.rows / downsample; ++pixel)
	{
		sample_info_downsampled.training_data_cx_cy.at<float>(pixel, 0)		= training_samples.training_data_cx_cy.at<float>(pixel * downsample, 0);
		sample_info_downsampled.training_data_cx_cy.at<float>(pixel, 1)		= training_samples.training_data_cx_cy.at<float>(pixel * downsample, 1);
		sample_info_downsampled.training_data_density.at<float>(pixel, 0)	= training_samples.training_data_density.at<float>(pixel * downsample, 0);
		sample_info_downsampled.class_data.at<float>(pixel, 0)				= training_samples.class_data.at<float>(pixel * downsample, 0);
	}

	return sample_info_downsampled;
}

NormalizedLutCreation::TransformationParameters NormalizedLutCreation::HandleParameterization(const TransformationParameters& calc_params, const boost::filesystem::path& template_file, const boost::filesystem::path& template_output, const size_t log_file_id)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	TransformationParameters lut_params = calc_params;
	if (!template_file.empty())
	{
		logging_instance->QueueFileLogging("Loading Template parameters...", log_file_id, IO::Logging::NORMAL);
		std::ifstream csv_input_stream;
		csv_input_stream.open(template_file.string());
		if (!csv_input_stream)
		{
			logging_instance->QueueCommandLineLogging("Could not read CSV file!", IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Could not read template CSV file!", log_file_id, IO::Logging::NORMAL);

			throw std::runtime_error("Could not read CSV file.");
		}
		else
		{
			lut_params = ReadParameters(csv_input_stream);
			csv_input_stream.close();
		}
	}

	if (!template_output.empty())
	{
		logging_instance->QueueFileLogging("Saving Template parameters...", log_file_id, IO::Logging::NORMAL);
		std::ofstream csv_output_stream;
		csv_output_stream.open(template_output.string());
		if (csv_output_stream)
		{
			PrintParameters(csv_output_stream, calc_params, true);
			csv_output_stream.close();

			logging_instance->QueueCommandLineLogging("Done", IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Template Parameters written to: " + template_output.string(), log_file_id, IO::Logging::NORMAL);
		}
		else
		{
			logging_instance->QueueCommandLineLogging("Could not write template CSV file!", IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Could not write template CSV file!", log_file_id, IO::Logging::NORMAL);
		}
	}

	return lut_params;
}

std::vector<cv::Mat> NormalizedLutCreation::InitializeTransformation(
	const cv::Mat& training_cx_cy,
	const cv::Mat& lut_cx_cy,
	const cv::Mat& cx_cy_hema_rotated,
	const cv::Mat& cx_cy_eosin_rotated,
	const TransformationParameters& params,
	const TransformationParameters& transform_params,
	const ClassPixelIndices& class_pixel_indices)
{
	// Rotates the combined matrices for the hema and eosin classes.
	cv::Mat lut_hema_matrix, lut_eosin_matrix;
	TransformCxCyDensity::RotateCxCy(lut_cx_cy,
										lut_hema_matrix,
		params.hema_rotation_params.x_median,
		params.hema_rotation_params.y_median,
		params.hema_rotation_params.angle);
	TransformCxCyDensity::RotateCxCy(lut_cx_cy,
										lut_eosin_matrix,
		params.eosin_rotation_params.x_median,
		params.eosin_rotation_params.y_median,
		params.eosin_rotation_params.angle);

	cv::Mat adjusted_hema_params(TransformCxCyDensity::AdjustParamaterMinMax(lut_hema_matrix, params.hema_scale_params));
	cv::Mat adjusted_eosin_params(TransformCxCyDensity::AdjustParamaterMinMax(lut_eosin_matrix, params.eosin_scale_params));

	// Scales the rotated LUT_matrices
	TransformCxCyDensity::ScaleCxCyLUT(lut_hema_matrix,	lut_hema_matrix, adjusted_hema_params, transform_params.hema_scale_params);
	TransformCxCyDensity::ScaleCxCy(lut_eosin_matrix, lut_eosin_matrix,	adjusted_eosin_params, transform_params.eosin_scale_params);

	cv::Mat hema_matrix, eosin_matrix;
	TransformCxCyDensity::ScaleCxCy(cx_cy_hema_rotated, hema_matrix, adjusted_hema_params, transform_params.hema_scale_params);
	TransformCxCyDensity::ScaleCxCy(cx_cy_eosin_rotated, eosin_matrix, adjusted_eosin_params, transform_params.eosin_scale_params);

	// Scales the rotated calculated matrices.
/*	TransformCxCyDensity::ScaleCxCy(rotated_hema,
										rotated_hema,
										params.hema_scale_params,
										transform_params.hema_scale_params);
	TransformCxCyDensity::ScaleCxCy(rotated_eosin,
										rotated_eosin,
										params.eosin_scale_params,
										transform_params.eosin_scale_params);
										*/
	// Reverses the rotation.
	TransformCxCyDensity::RotateCxCyBack(hema_matrix, hema_matrix, params.hema_rotation_params.angle);
	TransformCxCyDensity::RotateCxCyBack(eosin_matrix, eosin_matrix, params.eosin_rotation_params.angle);

	TransformCxCyDensity::RotateCxCyBack(lut_hema_matrix, lut_hema_matrix, transform_params.hema_rotation_params.angle - M_PI);
	TransformCxCyDensity::RotateCxCyBack(lut_eosin_matrix, lut_eosin_matrix, transform_params.eosin_rotation_params.angle - M_PI);

	cv::Mat lut_background_matrix;
	TransformCxCyDensity::TranslateCxCyBack(hema_matrix, lut_hema_matrix, lut_hema_matrix, class_pixel_indices.hema_indices, transform_params.hema_rotation_params.x_median, transform_params.hema_rotation_params.y_median);
	TransformCxCyDensity::TranslateCxCyBack(eosin_matrix, lut_eosin_matrix, lut_eosin_matrix, class_pixel_indices.eosin_indices, transform_params.eosin_rotation_params.x_median, transform_params.eosin_rotation_params.y_median);
	TransformCxCyDensity::TranslateCxCyBack(training_cx_cy, lut_cx_cy, lut_background_matrix, class_pixel_indices.background_indices, transform_params.background_rotation_params.x_median, transform_params.background_rotation_params.y_median);
	
	return { lut_hema_matrix, lut_eosin_matrix, lut_background_matrix };
}

void NormalizedLutCreation::PrintParameters(std::ofstream& output_stream, const TransformationParameters& transform_param, const bool write_csv)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	logging_instance->QueueCommandLineLogging("\nHematoxylin CxCy: "	+ std::to_string(transform_param.hema_rotation_params.x_median)			+ ", " + std::to_string(transform_param.hema_rotation_params.y_median)		+ "\n", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Eosin CxCy: "			+ std::to_string(transform_param.eosin_rotation_params.x_median)		+ ", " + std::to_string(transform_param.eosin_rotation_params.y_median)		+ "\n", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Background CxCy: "		+ std::to_string(transform_param.background_rotation_params.x_median)	+ ", " + std::to_string(transform_param.background_rotation_params.y_median)	+ "\n", IO::Logging::NORMAL);

	const cv::Mat& hema_scale_param(transform_param.hema_scale_params);
	const cv::Mat& eosin_scale_param(transform_param.eosin_scale_params);

	logging_instance->QueueCommandLineLogging("Hema Cx values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(hema_scale_param.at<float>(0, 0)) + " " + std::to_string(hema_scale_param.at<float>(1, 0)) + " " + std::to_string(hema_scale_param.at<float>(2, 0)) + " " +
		std::to_string(hema_scale_param.at<float>(3, 0)) + " " + std::to_string(hema_scale_param.at<float>(4, 0)) + " " + std::to_string(hema_scale_param.at<float>(5, 0)) + " " + std::to_string(hema_scale_param.at<float>(6, 0)), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\nHema Cy values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(hema_scale_param.at<float>(0, 1)) + " " + std::to_string(hema_scale_param.at<float>(1, 1)) + " " + std::to_string(hema_scale_param.at<float>(2, 1)) + " " +
		std::to_string(hema_scale_param.at<float>(3, 1)) + " " + std::to_string(hema_scale_param.at<float>(4, 1)) + " " + std::to_string(hema_scale_param.at<float>(5, 1)) + " " + std::to_string(hema_scale_param.at<float>(6, 1)), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\nEosin Cx values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(eosin_scale_param.at<float>(0, 0)) + " " + std::to_string(eosin_scale_param.at<float>(1, 0)) + " " + std::to_string(eosin_scale_param.at<float>(2, 0)) + " " +
		std::to_string(eosin_scale_param.at<float>(3, 0)) + " " + std::to_string(eosin_scale_param.at<float>(4, 0)) + " " + std::to_string(eosin_scale_param.at<float>(5, 0)) + " " + std::to_string(eosin_scale_param.at<float>(6, 0)), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\nEosin Cy values", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(std::to_string(eosin_scale_param.at<float>(0, 1)) + " " + std::to_string(eosin_scale_param.at<float>(1, 1)) + " " + std::to_string(eosin_scale_param.at<float>(2, 1)) + " " +
		std::to_string(eosin_scale_param.at<float>(3, 1)) + " " + std::to_string(eosin_scale_param.at<float>(4, 1)) + " " + std::to_string(eosin_scale_param.at<float>(5, 1)) + " " + std::to_string(eosin_scale_param.at<float>(6, 1)), IO::Logging::NORMAL);


	logging_instance->QueueCommandLineLogging("Hema angle: " + std::to_string(transform_param.hema_rotation_params.angle), IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Eos angle: " + std::to_string(transform_param.eosin_rotation_params.angle), IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Background angle: " + std::to_string(transform_param.background_rotation_params.angle), IO::Logging::NORMAL);

	logging_instance->QueueCommandLineLogging("\n\nH&E Density Params: ", IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging(
		std::to_string(transform_param.class_density_ranges.hema_density_mean.val[0]) + ", " + std::to_string(transform_param.class_density_ranges.hema_density_standard_deviation.val[0]) + ", " +
		std::to_string(transform_param.class_density_ranges.eosin_density_mean.val[0]) + ", " + std::to_string(transform_param.class_density_ranges.eosin_density_standard_deviation.val[0]) + ", " +
		std::to_string(transform_param.class_density_ranges.background_density_mean.val[0]) + ", " + std::to_string(transform_param.class_density_ranges.background_density_standard_deviation.val[0]),
		IO::Logging::NORMAL);

	if (write_csv)
	{
		output_stream << transform_param.hema_rotation_params.x_median			<< "," << transform_param.hema_rotation_params.y_median			<< std::endl;
		output_stream << transform_param.eosin_rotation_params.x_median			<< "," << transform_param.eosin_rotation_params.y_median		<< std::endl;
		output_stream << transform_param.background_rotation_params.x_median	<< "," << transform_param.background_rotation_params.y_median	<< std::endl;

		output_stream << hema_scale_param.at<float>(0, 0) << ", " << hema_scale_param.at<float>(1, 0) << ", " << hema_scale_param.at<float>(2, 0) << ", " << hema_scale_param.at<float>(3, 0) << ", " <<
			hema_scale_param.at<float>(4, 0) << ", " << hema_scale_param.at<float>(5, 0) << ", " << hema_scale_param.at<float>(6, 0) << std::endl;
		output_stream << hema_scale_param.at<float>(0, 1) << ", " << hema_scale_param.at<float>(1, 1) << ", " << hema_scale_param.at<float>(2, 1) << ", " << hema_scale_param.at<float>(3, 1) << ", " <<
			hema_scale_param.at<float>(4, 1) << ", " << hema_scale_param.at<float>(5, 1) << ", " << hema_scale_param.at<float>(6, 1) << std::endl;

		output_stream << eosin_scale_param.at<float>(0, 0) << ", " << eosin_scale_param.at<float>(1, 0) << ", " << eosin_scale_param.at<float>(2, 0) << ", " << eosin_scale_param.at<float>(3, 0) << ", " <<
			eosin_scale_param.at<float>(4, 0) << ", " << eosin_scale_param.at<float>(5, 0) << ", " << eosin_scale_param.at<float>(6, 0) << std::endl;
		output_stream << eosin_scale_param.at<float>(0, 1) << ", " << eosin_scale_param.at<float>(1, 1) << ", " << eosin_scale_param.at<float>(2, 1) << ", " << eosin_scale_param.at<float>(3, 1) << ", " <<
			eosin_scale_param.at<float>(4, 1) << ", " << eosin_scale_param.at<float>(5, 1) << ", " << eosin_scale_param.at<float>(6, 1) << std::endl;

		output_stream << transform_param.hema_rotation_params.angle << std::endl;
		output_stream << transform_param.eosin_rotation_params.angle << std::endl;
		output_stream << transform_param.background_rotation_params.angle << std::endl;
	
		output_stream <<	transform_param.class_density_ranges.hema_density_mean.val[0]		<< ", " << transform_param.class_density_ranges.hema_density_standard_deviation.val[0]			<< ", " <<
							transform_param.class_density_ranges.eosin_density_mean.val[0]		<< ", " << transform_param.class_density_ranges.eosin_density_standard_deviation.val[0]			<< ", " <<
							transform_param.class_density_ranges.background_density_mean.val[0] << ", " << transform_param.class_density_ranges.background_density_standard_deviation.val[0]	<< std::endl;
	}
}

NormalizedLutCreation::TransformationParameters NormalizedLutCreation::ReadParameters(std::istream &input)
{
	std::vector<std::vector<float>> csv_parameters;

	std::string csv_line;
	// read every line from the stream
	int i = 0;
	while (std::getline(input, csv_line))
	{
		int j = 0;
		std::istringstream csvStream(csv_line);
		std::vector<float> csvColumn;
		std::string csvElement;
		std::vector<float> rowvals;
		std::vector<std::string> all_words;
		// read every element from the line that is seperated by commas
		// and put it into the vector or strings
		while (std::getline(csvStream, csvElement, ','))
		{
			rowvals.push_back(atof(csvElement.c_str()));
			j++;
		}
		csv_parameters.push_back(rowvals);
		i++;
	}

	TransformationParameters parameters;

	parameters.hema_rotation_params.x_median = csv_parameters[0][0];
	parameters.hema_rotation_params.y_median = csv_parameters[0][1];

	parameters.eosin_rotation_params.x_median = csv_parameters[1][0];
	parameters.eosin_rotation_params.y_median = csv_parameters[1][1];

	parameters.background_rotation_params.x_median = csv_parameters[2][0];
	parameters.background_rotation_params.y_median = csv_parameters[2][1];

	for (size_t i = 0; i < 7; ++i)
	{
		parameters.hema_scale_params.at<float>(i, 0)	= csv_parameters[3][i];
		parameters.hema_scale_params.at<float>(i, 1)	= csv_parameters[4][i];

		parameters.eosin_scale_params.at<float>(i, 0)	= csv_parameters[5][i];
		parameters.eosin_scale_params.at<float>(i, 1)	= csv_parameters[6][i];
	}

	parameters.hema_rotation_params.angle		= csv_parameters[7][0];
	parameters.eosin_rotation_params.angle		= csv_parameters[8][0];
	parameters.background_rotation_params.angle = csv_parameters[9][0];

	parameters.class_density_ranges.hema_density_mean.val[0]						= csv_parameters[10][0];
	parameters.class_density_ranges.hema_density_standard_deviation.val[0]			= csv_parameters[10][1];
	parameters.class_density_ranges.eosin_density_mean.val[0]						= csv_parameters[10][2];
	parameters.class_density_ranges.eosin_density_standard_deviation.val[0]			= csv_parameters[10][3];
	parameters.class_density_ranges.background_density_mean.val[0]					= csv_parameters[10][4];
	parameters.class_density_ranges.background_density_standard_deviation.val[0]	= csv_parameters[10][5];

	return parameters;
}