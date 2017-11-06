#include "HSDmodel.h"
#include <iostream>

using namespace cv;
// Calculating HSD color model


HSDmodel::~HSDmodel()
{
	Red_density.release();
	Green_density.release();
	Blue_density.release();
	Density.release();
	Cx.release();
	Cy.release();
	NormalizedImage.release();
}


void HSDmodel::calculateHSD(Mat& image)
{
	Mat Blue, Green, Red;
	std::vector<Mat> channels(3);
	split(image, channels);

	// Build HSD model
	Blue = channels[0];
	Green = channels[1];
	Red = channels[2];

	for(int row = 0; row < Red.rows; ++row) {
		uchar* red_Rowp = Red.ptr(row);
		uchar* blue_Rowp = Blue.ptr(row);
		uchar* green_Rowp = Green.ptr(row);
		for(int col = 0; col < Red.cols; ++col) {
			 if (*red_Rowp == 0)
				 *red_Rowp = 1;
			 if (*red_Rowp == 255)
				 *red_Rowp = 254;
			 if (*blue_Rowp == 0)
				 *blue_Rowp = 1;
			 if (*blue_Rowp == 255)
				 *blue_Rowp = 254;
			 if (*green_Rowp == 0)
				 *green_Rowp = 1;
			 if (*green_Rowp == 255)
				 *green_Rowp = 254;
			 
			 *red_Rowp++;  //points to each pixel value in turn assuming a CV_8UC1 greyscale image 
			 *blue_Rowp++;
			 *green_Rowp++;
		}
	}

	Red.convertTo(Red,CV_32F);
	Green.convertTo(Green,CV_32F);
	Blue.convertTo(Blue,CV_32F);
	Red_density = Red / 255;
	Blue_density = Blue / 255;
	Green_density = Green / 255;

	log(Red_density,Red_density);
	log(Green_density,Green_density);
	log(Blue_density,Blue_density);

	Red_density *= -1;
	Green_density *= -1;
	Blue_density *= -1;

	//convertScaleAbs(Blue_density,Blue_density);
	Density = (Red_density+Green_density+Blue_density)/3;
	Cx = Red_density / Density - 1;
	Cy = (Green_density - Blue_density)/(sqrt(3.) * Density);
}


void HSDmodel::calculateHSD2(Mat& image)
{
	Mat Blue, Green, Red;
	std::vector<Mat> channels(3);
	split(image, channels);

	// Build HSD model
	Blue = channels[2];
	Green = channels[1];
	Red = channels[0];

	for(int row = 0; row < Red.rows; ++row) {
		uchar* red_Rowp = Red.ptr(row);
		uchar* blue_Rowp = Blue.ptr(row);
		uchar* green_Rowp = Green.ptr(row);
		for(int col = 0; col < Red.cols; ++col) {
			 if (*red_Rowp == 0)
				 *red_Rowp = 1;
			 if (*red_Rowp == 255)
				 *red_Rowp = 254;
			 if (*blue_Rowp == 0)
				 *blue_Rowp = 1;
			 if (*blue_Rowp == 255)
				 *blue_Rowp = 254;
			 if (*green_Rowp == 0)
				 *green_Rowp = 1;
			 if (*green_Rowp == 255)
				 *green_Rowp = 254;
			 
			 *red_Rowp++;  //points to each pixel value in turn assuming a CV_8UC1 greyscale image 
			 *blue_Rowp++;
			 *green_Rowp++;
		}
	}

	Red.convertTo(Red,CV_32F);
	Green.convertTo(Green,CV_32F);
	Blue.convertTo(Blue,CV_32F);
	Red_density = Red / 255;
	Blue_density = Blue / 255;
	Green_density = Green / 255;

	log(Red_density,Red_density);
	log(Green_density,Green_density);
	log(Blue_density,Blue_density);

	Red_density *= -1;
	Green_density *= -1;
	Blue_density *= -1;

	//convertScaleAbs(Blue_density,Blue_density);
	Density = (Red_density+Green_density+Blue_density)/3;
	Cx = Red_density / Density - 1;
	Cy = (Green_density - Blue_density)/(sqrt(3.) * Density);
}



// Calculating HSD reverse transform to get bak to RGB color model
void HSDmodel::HSDreverse(Mat& Scaled_DensityIamge, Mat& Cx_imgNorm, Mat& Cy_imgNorm)
{
	Mat D_red = Scaled_DensityIamge.mul(Cx_imgNorm + 1);
	Mat D_green = 0.5 * Scaled_DensityIamge.mul(2 - Cx_imgNorm + sqrt(3.) * Cy_imgNorm);
	Mat D_blue = 0.5 * Scaled_DensityIamge.mul(2 - Cx_imgNorm - sqrt(3.) * Cy_imgNorm);
	
	Mat I_red = Mat::zeros(Scaled_DensityIamge.size() , CV_32FC1); 
	Mat I_green = Mat::zeros(Scaled_DensityIamge.size() , CV_32FC1); 
	Mat I_blue = Mat::zeros(Scaled_DensityIamge.size() , CV_32FC1);
	
	for (int i = 0; i < Scaled_DensityIamge.rows; ++i)
	{
		for (int j = 0; j < Scaled_DensityIamge.cols; ++j)
		{
			I_red.at<float>(i, j) = 255 * exp(-D_red.at<float>(i, j));
			I_green.at<float>(i, j) = 255 * exp(-D_green.at<float>(i, j));
			I_blue.at<float>(i, j) = 255 * exp(-D_blue.at<float>(i, j));
		}
	}
	std::vector<Mat> channels;
	channels.push_back(I_blue);
	channels.push_back(I_green);
	channels.push_back(I_red);
	merge(channels, NormalizedImage);
}


void HSDmodel::HSDreverse(Mat& Cx_imgNorm, Mat& Cy_imgNorm)
{
	Mat D_red = 0.2 *(Cx_imgNorm + 1);
	Mat D_green = 0.5 *0.2*(2 - Cx_imgNorm + sqrt(3.) * Cy_imgNorm);
	Mat D_blue = 0.5 * 0.2*(2 - Cx_imgNorm - sqrt(3.) * Cy_imgNorm);
	
	Mat I_red = Mat::zeros(Cx_imgNorm.size() , CV_32FC1); 
	Mat I_green = Mat::zeros(Cx_imgNorm.size() , CV_32FC1); 
	Mat I_blue = Mat::zeros(Cx_imgNorm.size() , CV_32FC1);
	
	for (int i = 0; i < Cx_imgNorm.rows; ++i)
	{
		for (int j = 0; j < Cx_imgNorm.cols; ++j)
		{
			I_red.at<float>(i, j) = 255 * exp(-D_red.at<float>(i, j));
			I_green.at<float>(i, j) = 255 * exp(-D_green.at<float>(i, j));
			I_blue.at<float>(i, j) = 255 * exp(-D_blue.at<float>(i, j));
		}
	}
	std::cout << "RGB ready..." << std::endl;
	std::vector<Mat> channels;
	channels.push_back(I_blue);
	channels.push_back(I_green);
	channels.push_back(I_red);
	merge(channels, NormalizedImage);
	NormalizedImage.convertTo(NormalizedImage,CV_8UC3);
}