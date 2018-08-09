#include "PixelClassificationHE.h"

#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>

#include "../HSD/BackgroundMask.h"
#include "../IO/Logging/LogHandler.h"
#include "../Misc/LevelReading.h"
#include "../Misc/MiscFunctionality.h"

// TODO: Improve structure and refactor InsertTrainingData_

PixelClassificationHE::PixelClassificationHE(const bool consider_ink, const size_t log_file_id, const std::string debug_dir) : m_consider_ink_(consider_ink), m_log_file_id_(log_file_id), m_debug_dir_(debug_dir)
{
}

TrainingSampleInformation PixelClassificationHE::GenerateCxCyDSamples(
	MultiResolutionImage& tiled_image,
	const cv::Mat& static_image,
	const StandardizationParameters& parameters,
	const std::vector<cv::Point>& tile_coordinates,
	const std::vector<double>& spacing,
	const uint32_t tile_size,
	const uint32_t min_level,
	const bool is_multiresolution_image)
{
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());

	uint32_t min_training_size = 0;
	if (is_multiresolution_image)
	{
		min_training_size = parameters.min_training_size;
	}

	logging_instance->QueueCommandLineLogging("Minimum number of samples to take from the WSI: " + std::to_string(parameters.max_training_size), IO::Logging::NORMAL);
	logging_instance->QueueCommandLineLogging("Minimum number of samples to take from each patch: " + std::to_string(parameters.min_training_size), IO::Logging::NORMAL);

	// Tracks the created samples for each class.
	size_t total_hema_count = 0;
	size_t total_eosin_count = 0;
	size_t total_background_count = 0;

	TrainingSampleInformation sample_information{ cv::Mat::zeros(parameters.max_training_size, 2, CV_32FC1), cv::Mat::zeros(parameters.max_training_size, 1, CV_32FC1), cv::Mat::zeros(parameters.max_training_size, 1, CV_32FC1) };

	size_t selected_images_count = 0;
	std::vector<size_t> random_numbers(ASAP::MiscFunctionality::CreateListOfRandomIntegers(tile_coordinates.size()));
	for (size_t current_tile = 0; current_tile < tile_coordinates.size(); ++current_tile)
	{
		logging_instance->QueueCommandLineLogging(std::to_string(current_tile + 1) + " images taken as examples!", IO::Logging::NORMAL);
		logging_instance->QueueFileLogging("=============================\nRandom Tile " + std::to_string(random_numbers[current_tile] + 1) + "\n=============================", m_log_file_id_, IO::Logging::NORMAL);

		//===========================================================================
		//	HSD / CxCy Color Model
		//===========================================================================
		HSD::HSD_Model hsd_image;
		if (is_multiresolution_image)
		{
			unsigned char* data(nullptr);

			tiled_image.getRawRegion(tile_coordinates[random_numbers[current_tile]].x * tiled_image.getLevelDownsample(0),
				tile_coordinates[random_numbers[current_tile]].y * tiled_image.getLevelDownsample(0),
				tile_size, tile_size, min_level, data);

			cv::Mat raw_image = cv::Mat::zeros(tile_size, tile_size, CV_8UC3);
			LevelReading::ArrayToMatrix(data, raw_image, false);
			if (IO::Logging::LogHandler::GetInstance()->GetOutputLevel() == IO::Logging::DEBUG && !m_debug_dir_.empty())
			{
				std::string original_name(m_debug_dir_ + "/tile_" + std::to_string(random_numbers[current_tile]) + "_raw.tif");
				cv::imwrite(original_name, raw_image);
			}
			hsd_image = HSD::HSD_Model(raw_image, HSD::BGR);
		}
		else
		{
			hsd_image = HSD::HSD_Model(static_image, HSD::BGR);
		}

		//===========================================================================
		//	Background Mask
		//===========================================================================
		cv::Mat background_mask(HSD::BackgroundMask::CreateBackgroundMask(hsd_image, 0.24, 0.22));

		//*************************************************************************
		// Sample extraction with Hough Transform
		//*************************************************************************
		// Attempts to acquire the HE stain masks, followed by the classification of the image. Which results in tissue, class, train and test data.
		std::pair<HematoxylinMaskInformation, EosinMaskInformation> he_masks(Create_HE_Masks_(hsd_image,
			background_mask,
			random_numbers[current_tile],
			min_training_size,
			parameters.minimum_ellipses,
			parameters.hema_percentile,
			parameters.eosin_percentile,
			spacing,
			is_multiresolution_image));
		HE_Staining::HE_Classifier he_classifier;
		HE_Staining::ClassificationResults classification_results(he_classifier.Classify(hsd_image, background_mask, he_masks.first, he_masks.second));

		// Wanna keep?
		// Randomly pick samples and Fill in Cx-Cy-D major sample vectors	
		if (classification_results.train_and_class_data.train_data.rows >= min_training_size)
		{
			if (logging_instance->GetOutputLevel() == IO::Logging::DEBUG && !m_debug_dir_.empty())
			{
				cv::imwrite(m_debug_dir_ + "/tile_" + std::to_string(random_numbers[current_tile]) + "_classified.tif", classification_results.all_classes * 100);

				std::vector<cv::Mat> channels;
				channels.push_back(he_masks.second.full_mask);
				channels.push_back(background_mask);
				channels.push_back(he_masks.first.full_mask);

				cv::Mat classes;
				cv::merge(channels, classes);
				classes.convertTo(classes, CV_8UC1);
				cv::imwrite(m_debug_dir_ + "/tile_" + std::to_string(random_numbers[current_tile]) + "_classes.tif", classes * 100);
			}


		//	TrainingSampleInformation& sample_information, const cv::Mat& cx_cy, const cv::Mat& density, const std::vector<cv::Point>& data_pixels, const size_t insertion_start, const size_t pixels_to_insert, const uint8_t class_id


			InsertTrainingData_(hsd_image, classification_results, sample_information, total_hema_count, total_eosin_count, total_background_count, parameters.max_training_size);
			++selected_images_count;

			size_t hema_count_real = total_hema_count				> parameters.max_training_size * 9 / 20 ? hema_count_real		= parameters.max_training_size * 9 / 20 : total_hema_count;
			size_t eosin_count_real = total_eosin_count				> parameters.max_training_size * 9 / 20 ? eosin_count_real		= parameters.max_training_size * 9 / 20 : total_eosin_count;
			size_t background_count_real = total_background_count	> parameters.max_training_size * 1 / 10 ? background_count_real	= parameters.max_training_size * 1 / 10 : total_background_count;

			logging_instance->QueueCommandLineLogging(std::to_string(hema_count_real + eosin_count_real + background_count_real) + " training sets are filled, out of " + std::to_string(parameters.max_training_size) + " required.",	IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Filled: " + std::to_string(hema_count_real + eosin_count_real + background_count_real) + " / " + std::to_string(parameters.max_training_size),	m_log_file_id_,	IO::Logging::NORMAL);
			logging_instance->QueueFileLogging("Hema: " + std::to_string(hema_count_real) + ", Eos: " + std::to_string(eosin_count_real) + ", BG: " + std::to_string(background_count_real), m_log_file_id_, IO::Logging::NORMAL);
		}

		if (total_hema_count >= parameters.max_training_size * 9 / 20 && total_eosin_count >= parameters.max_training_size * 9 / 20 && total_background_count >= parameters.max_training_size / 10)
		{
			break;
		}
	}

	if ((total_hema_count < parameters.max_training_size * 9 / 20 || total_eosin_count < parameters.max_training_size * 9 / 20 || total_background_count < parameters.max_training_size / 10))
	{
		size_t non_zero_class_pixels = cv::countNonZero(sample_information.class_data);

		if (non_zero_class_pixels != sample_information.class_data.rows && (selected_images_count > 2 || !min_training_size))
		{
			std::string log_text("Could not fill all the " + std::to_string(parameters.max_training_size) + " samples required. Continuing with what is left...");
			logging_instance->QueueCommandLineLogging(log_text, IO::Logging::NORMAL);
			logging_instance->QueueFileLogging(log_text, m_log_file_id_, IO::Logging::NORMAL);

			sample_information = PatchTestData_(non_zero_class_pixels, sample_information);
		}
	}

	return sample_information;
}

std::pair<HematoxylinMaskInformation, EosinMaskInformation> PixelClassificationHE::Create_HE_Masks_(
	const HSD::HSD_Model& hsd_image,
	const cv::Mat& background_mask,
	const uint32_t tile_id,
	const uint32_t min_training_size,
	const int32_t min_ellipses,
	const float hema_percentile,
	const float eosin_percentile,
	const std::vector<double>& spacing,
	const bool is_multiresolution)
{
	// Sets the variables for the ellipse detection. 
	HoughTransform::RandomizedHoughTransformParameters parameters(HoughTransform::RandomizedHoughTransform::GetStandardParameters());
	parameters.min_ellipse_radius = floor(1.94 / spacing[0]);
	parameters.max_ellipse_radius = ceil(4.86 / spacing[0]);
	parameters.epoch_size = 3;
	parameters.count_threshold = 4;

	int sigma = 4;
	int low_threshold = 45;
	int high_threshold = 80;

	// Prepares the logger and a string holding potential failure messages.
	IO::Logging::LogHandler* logging_instance(IO::Logging::LogHandler::GetInstance());
	std::string failure_log_message;

	if (logging_instance->GetOutputLevel() == IO::Logging::DEBUG && !m_debug_dir_.empty())
	{
		cv::Mat temporary_matrix;
		hsd_image.density.convertTo(temporary_matrix, CV_8UC1);
		cv::imwrite(m_debug_dir_ + "/tile_" + std::to_string(tile_id) + "_density.tif", temporary_matrix * 100);

		HE_Staining::MaskGeneration::ApplyBlur(temporary_matrix, temporary_matrix, sigma);
		HE_Staining::MaskGeneration::ApplyCannyEdge(temporary_matrix, temporary_matrix, low_threshold, high_threshold);
		cv::imwrite(m_debug_dir_ + "/tile_" + std::to_string(tile_id) + "_binary.tif", temporary_matrix * 255);
	}

	// Attempts to detect ellispes. Applies a blur and canny edge operation on the density matrix before detecting ellipses through a randomized Hough transform.
	std::vector<HoughTransform::Ellipse> detected_ellipses(HE_Staining::MaskGeneration::DetectEllipses(hsd_image.density, sigma, low_threshold, high_threshold, parameters));

	logging_instance->QueueCommandLineLogging("Number of ellipses is: " + std::to_string(detected_ellipses.size()), IO::Logging::NORMAL);

	std::pair<HE_Staining::HematoxylinMaskInformation, HE_Staining::EosinMaskInformation> mask_acquisition_results;

	// If the min_ellipses variable has been set as positive, prefer it over a calculation.
	double min_detected_ellipses = min_ellipses > 0 ? hsd_image.red_density.rows * hsd_image.red_density.rows * spacing[0] * spacing[0] * (150 / 247700.0) : min_ellipses;
	if (detected_ellipses.size() > min_detected_ellipses || (detected_ellipses.size() > 10 && !is_multiresolution))
	{
		logging_instance->QueueFileLogging("Passed step 1: Number of nuclei " + std::to_string(detected_ellipses.size()), m_log_file_id_, IO::Logging::NORMAL);

		std::pair<bool, HE_Staining::HematoxylinMaskInformation> hema_mask_acquisition_result;
		if (!(hema_mask_acquisition_result = HE_Staining::MaskGeneration::GenerateHematoxylinMasks(hsd_image, background_mask, detected_ellipses, hema_percentile)).first || !m_consider_ink_)
		{
			logging_instance->QueueFileLogging("Skipped - May contain INK.", m_log_file_id_, IO::Logging::NORMAL);
		}

		// Creates a reference of the hema mask information, for ease of use. And copies the results into the result pairing.
		mask_acquisition_results.first = std::move(hema_mask_acquisition_result.second);
		HE_Staining::HematoxylinMaskInformation& hema_mask_info(mask_acquisition_results.first);

		// Test and Train data Generator
		if (hema_mask_info.training_pixels > min_training_size / 2)
		{
			logging_instance->QueueFileLogging(
				"Passed step 2: Amount of Hema samples " + std::to_string(hema_mask_info.training_pixels) + ", more than the threshold of " + std::to_string(min_training_size / 2),
				m_log_file_id_,
				IO::Logging::NORMAL);

			HE_Staining::EosinMaskInformation eosin_mask_info(HE_Staining::MaskGeneration::GenerateEosinMasks(hsd_image, background_mask, hema_mask_info, 0.85));
			if (eosin_mask_info.training_pixels > min_training_size / 2)
			{
				mask_acquisition_results.second = eosin_mask_info;

				IO::Logging::LogHandler::GetInstance()->QueueFileLogging(
					"Passed step 3: Amount of Eosin samples " + std::to_string(eosin_mask_info.training_pixels) + ", more than the threshold of " + std::to_string(min_training_size / 2),
					m_log_file_id_,
					IO::Logging::NORMAL);
			}
			else
			{
				failure_log_message = "Skipped step 3: Amount of Eosin samples " + std::to_string(eosin_mask_info.training_pixels) + ", less than the threshold of " + std::to_string(min_training_size / 2);
			}
		}
		else
		{
			failure_log_message = "Skipped step 2: Amount of Hema samples " + std::to_string(hema_mask_info.training_pixels) + ", less than the threshold of " + std::to_string(min_training_size / 2);
		}
	}
	else
	{
		failure_log_message = "Skipped: Nuclei " + std::to_string(detected_ellipses.size()) + " < " + std::to_string(min_detected_ellipses);
	}

	if (!failure_log_message.empty())
	{
		logging_instance->QueueCommandLineLogging(failure_log_message, IO::Logging::NORMAL);
		logging_instance->QueueFileLogging(failure_log_message, m_log_file_id_, IO::Logging::NORMAL);
	}

	return mask_acquisition_results;
}
	/*
void PixelClassificationHE::InsertTrainingData_(TrainingSampleInformation& sample_information, const cv::Mat& cx_cy, const cv::Mat& density, const std::vector<cv::Point>& data_pixels, const size_t insertion_start, const size_t pixels_to_insert, const uint8_t class_id)
{
	std::vector<size_t> random_pixels(ASAP::MiscFunctionality::CreateListOfRandomIntegers(data_pixels.size()));
	for (size_t pixel = 0; pixel < pixels_to_insert; ++pixel)
	{
		sample_information.training_data_cx_cy.at<float>(insertion_start + pixel, 0)	= cx_cy.at<float>(data_pixels[random_pixels[pixel]].y, 0);
		sample_information.training_data_cx_cy.at<float>(insertion_start + pixel, 1)	= cx_cy.at<float>(data_pixels[random_pixels[pixel]].y, 1);
		sample_information.training_data_density.at<float>(insertion_start + pixel, 0)	= density.at<float>((data_pixels[random_pixels[pixel]]));
		sample_information.class_data.at<float>(insertion_start + pixel, 0)				= class_id;

	}
}*/

TrainingSampleInformation PixelClassificationHE::InsertTrainingData_(
	const HSD::HSD_Model& hsd_image,
	const ClassificationResults& classification_results,
	TrainingSampleInformation& sample_information,
	size_t& total_hema_count,
	size_t& total_eosin_count,
	size_t& total_background_count,
	const uint32_t max_training_size)
{
	// Creates a list of random values, ranging from 0 to the amount of class pixels - 1.
	std::vector<size_t> hema_random_numbers(ASAP::MiscFunctionality::CreateListOfRandomIntegers(classification_results.hema_pixels));
	std::vector<size_t> eosin_random_numbers(ASAP::MiscFunctionality::CreateListOfRandomIntegers(classification_results.eosin_pixels));
	std::vector<size_t> background_random_numbers(ASAP::MiscFunctionality::CreateListOfRandomIntegers(classification_results.background_pixels));

	// Tracks the amount of class specific sample counts. And the matrices which will hold information for each pixel classified as their own.
	size_t local_hema_count = 0;
	size_t local_eosin_count = 0;
	size_t local_background_count = 0;
	cv::Mat train_data_hema(cv::Mat::zeros(classification_results.hema_pixels, 3, CV_32FC1));
	cv::Mat train_data_eosin(cv::Mat::zeros(classification_results.eosin_pixels, 3, CV_32FC1));
	cv::Mat train_data_background(cv::Mat::zeros(classification_results.background_pixels, 3, CV_32FC1));

	// Creates a matrix per class, holding the c_x, c_y and density channels per classified pixel.
	const cv::Mat& all_classes(classification_results.all_classes);
	for (int row = 0; row < all_classes.rows; ++row)
	{
		const uchar* Class = all_classes.ptr(row);
		for (int col = 0; col < all_classes.cols; ++col)
		{
			if (*Class == 1)
			{
				train_data_hema.at<float>(local_hema_count, 0) = hsd_image.c_x.at<float>(row, col);
				train_data_hema.at<float>(local_hema_count, 1) = hsd_image.c_y.at<float>(row, col);
				train_data_hema.at<float>(local_hema_count, 2) = hsd_image.density.at<float>(row, col);
				++local_hema_count;
			}
			else if (*Class == 2)
			{
				train_data_eosin.at<float>(local_eosin_count, 0) = hsd_image.c_x.at<float>(row, col);
				train_data_eosin.at<float>(local_eosin_count, 1) = hsd_image.c_y.at<float>(row, col);
				train_data_eosin.at<float>(local_eosin_count, 2) = hsd_image.density.at<float>(row, col);
				++local_eosin_count;
			}
			else if (*Class == 3)
			{
				train_data_background.at<float>(local_background_count, 0) = hsd_image.c_x.at<float>(row, col);
				train_data_background.at<float>(local_background_count, 1) = hsd_image.c_y.at<float>(row, col);
				train_data_background.at<float>(local_background_count, 2) = hsd_image.density.at<float>(row, col);
				++local_background_count;
			}
			*Class++;
		}
	}

	auto insert_sample_information = [](size_t class_pixel, size_t training_pixel, char class_id, TrainingSampleInformation& sample_information, cv::Mat& class_matrix)
	{
		sample_information.training_data_cx_cy.at<float>(training_pixel, 0) = class_matrix.at<float>(class_pixel, 0);
		sample_information.training_data_cx_cy.at<float>(training_pixel, 1) = class_matrix.at<float>(class_pixel, 1);
		sample_information.training_data_density.at<float>(training_pixel, 0) = class_matrix.at<float>(class_pixel, 2);
		sample_information.class_data.at<float>(training_pixel, 0) = class_id;
	};

	for (size_t hema_pixel = 0; hema_pixel < classification_results.hema_pixels / 2; ++hema_pixel)
	{
		size_t training_pixel = hema_pixel + total_hema_count;
		if (training_pixel < max_training_size * 9 / 20)
		{
			insert_sample_information(hema_random_numbers[hema_pixel], training_pixel, 1, sample_information, train_data_hema);
		}
	}
	for (size_t eosin_pixel = 0; eosin_pixel < classification_results.eosin_pixels / 2; ++eosin_pixel)
	{
		size_t training_pixel = eosin_pixel + total_eosin_count + max_training_size * 9 / 20;
		if (training_pixel < max_training_size * 18 / 20)
		{
			insert_sample_information(eosin_random_numbers[eosin_pixel], training_pixel, 2, sample_information, train_data_eosin);
		}
	}
	for (size_t background_pixel = 0; background_pixel < classification_results.background_pixels / 2; ++background_pixel)
	{
		size_t training_pixel = background_pixel + total_background_count + max_training_size * 18 / 20;
		if (training_pixel < max_training_size)
		{
			insert_sample_information(background_random_numbers[background_pixel], training_pixel, 3, sample_information, train_data_background);
		}
	}

	// Adds the local counts to the total count.
	total_hema_count += local_hema_count / 2;
	total_eosin_count += local_eosin_count / 2;
	total_background_count += local_background_count / 2;

	return sample_information;
}

TrainingSampleInformation PixelClassificationHE::PatchTestData_(const size_t non_zero_count, const TrainingSampleInformation& current_sample_information)
{
	TrainingSampleInformation new_sample_information{ cv::Mat::zeros(non_zero_count, 2, CV_32FC1), cv::Mat::zeros(non_zero_count, 1, CV_32FC1),	cv::Mat::zeros(non_zero_count, 1, CV_32FC1) };

	size_t added_samples_count = 0;
	for (size_t count = 0; count < current_sample_information.class_data.rows; ++count)
	{
		if (current_sample_information.class_data.at<float>(count, 0) != 0)
		{
			new_sample_information.training_data_cx_cy.at<float>(added_samples_count, 0)	= current_sample_information.training_data_cx_cy.at<float>(count, 0);
			new_sample_information.training_data_cx_cy.at<float>(added_samples_count, 1)	= current_sample_information.training_data_cx_cy.at<float>(count, 1);
			new_sample_information.training_data_density.at<float>(added_samples_count, 0)	= current_sample_information.training_data_density.at<float>(count, 0);
			new_sample_information.class_data.at<float>(added_samples_count, 0)				= current_sample_information.class_data.at<float>(count, 0);
			++added_samples_count;
		}
	}

	return new_sample_information;
}