#ifndef __HSD_HSD_Model_H__
#define __HSD_HSD_Model_H__

#include "opencv2/core.hpp"

namespace HSD
{
	/// <summary>
	/// Defines the way a HSD_Model object is initialized. Whether or not to retain the color channel order, or to shift it.
	/// </summary>
	enum HSD_Initialization_Type { RGB, BGR };

	/// <summary>
	/// Represents an image in the HSD model.
	/// </summary>
	class HSD_Model
	{
		public:
			cv::Mat red_density, green_density, blue_density;
			cv::Mat density;
			cv::Mat c_x;
			cv::Mat c_y;

			/// <summary>
			/// Initializes the HSD_Model container through the conversion of a RGB image.
			/// </summary>
			/// <param name="rgb_image">A matrix containing the RGB representation of an image.</param>
			/// <param name="initialization_type">The type of initialization to use, which defines the order of the color channels.</param>
			HSD_Model(const cv::Mat& rgb_image, const HSD_Initialization_Type initialization_type);

			/// <summary>
			/// Returns the type of initialization used for this object.
			/// </summary>
			/// <returns>A HSD_Initialization_Type, which defines the type of initialization used.</returns>
			HSD_Initialization_Type GetInitializationType(void) const;

		private:
			HSD_Initialization_Type m_initialization_type_;

			/// <summary>
			/// Converts one of the RGB color channel matrices to the HSD model.
			/// </summary>
			/// <param name="matrix">A color channel matrix to convert to the HSD model.</param>
			void AdjustChannelMatrix_(cv::Mat& matrix);
	};
}
#endif // __HSD_HSD_Model_H__