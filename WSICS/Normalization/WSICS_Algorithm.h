#ifndef __WSICS_NORMALIZATION_WSICSALGORITHM__
#define __WSICS_NORMALIZATION_WSICSALGORITHM__

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include "PixelClassificationHE.h"
#include "WSICS_Parameters.h"
#include "TransformCxCyDensity.h"

namespace WSICS::Normalization
{
	class WSICS_Algorithm
	{
		public:
			WSICS_Algorithm(std::string log_directory, const boost::filesystem::path& template_file);
			WSICS_Algorithm(std::string log_directory, const boost::filesystem::path& template_file, const WSICS_Parameters& parameters);

			static WSICS_Parameters GetStandardParameters(void);
			void Normalize(
				const boost::filesystem::path& input_file,
				const boost::filesystem::path& image_output_file,
				const boost::filesystem::path& lut_output_file,
				const boost::filesystem::path& template_output_file,
				const boost::filesystem::path& debug_directory);
			void SetLogDirectory(std::string& log_directory);

		private:
			size_t							m_log_file_id_;
			boost::filesystem::path			m_debug_directory_;
			const boost::filesystem::path&	m_template_file_;

			WSICS_Parameters				m_parameters_;
			bool							m_is_multiresolution_image_;

			cv::Mat									CalculateLutRawMat_(void);
			std::pair<bool, std::vector<double>>	GetResolutionTypeAndSpacing(MultiResolutionImage& tiled_image);
			std::vector<cv::Point>					GetTileCoordinates_(MultiResolutionImage& tiled_image, const std::vector<double>& spacing, const uint32_t tile_size, const uint32_t min_level);

			TrainingSampleInformation CollectTrainingSamples_(
				const boost::filesystem::path& input_file,
				uint32_t tile_size,
				MultiResolutionImage& tiled_image,
				cv::Mat static_image,
				const std::vector<cv::Point>& tile_coordinates,
				const std::vector<double>& spacing,
				const uint32_t min_level);
	};
}
#endif // __WSICS_NORMALIZATION_WSICSALGORITHM__