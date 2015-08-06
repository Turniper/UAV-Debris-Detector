#ifndef PREP_IMG_H
#define PREP_IMG_H


#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// Global Variables
int NAME_SIZE = 42;

// Function headers
Mat transformImage(Mat src);
Mat bilateralBlur(Mat src);
Mat equalizeIntensity(const Mat& inputImage);
void display(Mat src, Mat dst);


/*
transformImage function
Purpose:       Apply bilateral blur and histogram equalization to the
image and return it
Precondition:  The original image has been loaded and stored in src.
Postcondition: The image was downsampled, bilateral blur and histogram
equalization were applied to it, and the result was returned to the
calling function.
*/
Mat transformImage(Mat src)
{
	Mat result = bilateralBlur(src);
	result = equalizeIntensity(result);

	return result;
}

/*
bilateralBlur function
Purpose:       Apply bilateral blur to the image and return it
Precondition:  The original image has been loaded and stored in src.
Postcondition: Bilateral blur was applied to the image stored in src
returned to the calling function.
*/
Mat bilateralBlur(Mat src)
{
	Mat dst,
		mst,
		lst;

	// Might need to lower value of d to process faster
	int d = 15;

	// Downsample twice for efficiency purposes
	resize(src, mst, Size(int(src.cols / 4), int(src.rows / 4)));

	bilateralFilter(mst, lst, d, 62, 15);

	return lst;
}

/*
equalizeIntensity function
Purpose:       Equalize intensity of an image
Precondition:  The inputImage has been loaded.
Postcondition: An image of the inputImage with equalized
intensity has been returned.
*/
Mat equalizeIntensity(const Mat& inputImage)
{
	if (inputImage.channels() >= 3)
	{
		Mat ycrcb;
		cvtColor(inputImage, ycrcb, CV_BGR2YCrCb);

		vector<Mat> channels;
		split(ycrcb, channels);

		equalizeHist(channels[0], channels[0]);

		Mat result;
		merge(channels, ycrcb);
		cvtColor(ycrcb, result, CV_YCrCb2BGR);

		return result;
	}

	return Mat();
}

/*
display function
Purpose:       Display an image to the user
Precondition:  An image is contained in pic.
Postcondition: The image stored in pic has been displayed.
*/
void display(Mat src, Mat dst)
{
	namedWindow("Source Image", 0);
	resizeWindow("Source Image", 528 + 264, 297 + 149);
	imshow("Source Image", src);

	namedWindow("Transformed Image", 0);
	resizeWindow("Transformed Image", 528 + 264, 297 + 149);
	//	cvNamedWindow("Transformed Image", CV_WINDOW_NORMAL);
	imshow("Transformed Image", dst);
}

#endif
