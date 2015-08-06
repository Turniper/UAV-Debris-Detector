//Matt Hegarty
//6/25/2015
//Texas A&M Department of Computer Science and Engineering
//A program to detect and flag stills from UAV imagery based on the presence of urban debris
//This program relies heavily on shape and color detection in it's approach to feature detection.
//--------------------------------------------------------------------------
//Debug and dev controls are triggered by the ints debug and dbdraw.  Debug lets you run tests and use the other modes,
//dbdraw shows a visual representation of whats going on during processing.
//--------------------------------------------------------------------------
//Dependencies:
//OpenCV 3.0
//Tinydir is also a dependency, but it is included with this repository.
//--------------------------------------------------------------------------
//A paper explaining the general workings of this tool is linked from the repository,
//the code itself is 
//--------------------------------------------------------------------------
/* No coding project is complete without ascii pictures of kittens.
    /\__/\					
   /`    '\					
 === 0  0 ===				
   \  --  /					   ,/|          _.--''^``-...___.._.,;
  /        \				  /, \'.     _-'          ,--,,,--''
 /          \				 { \    `_-''       '    /}
|            |				  `;;'              ;   ; ;
 \  ||  ||  /				  ._.--''     ._,,, _..'  .;.'
  \_oo__oo_/#######O		   (,_....----'''     (,..--''
//--------------------------------------------------------------------------
Serious usage information.
The main method contains several additional options.  (The int mode)
1-4 are for testing different portions, 5 provides the actual functionality of the application,
and 7 is a test mode for when a set of data has already been classified.  It's currently set to just run
mode 5, which takes the path to a directory of images and places flagged ones in a new folder.

There are several function stubs, they are denoted with //STUB



//---------------------------------------------------------------------------
//Things to implement and the combinations I want to test them in.

//Corner Detection Algorithms--------------
Shi-Tomasi - Consider shi-tomasi ignoring points with no other corners within 100 pixels in any direction.
Tested Harris, S-T seems to work a lot better for this application

//Filters----------------------------------
Mathematical Morphology - Experiment with more complicated operations than just erosion and dilation

//Color Detection--------------------------
HSV conversion + Segmentation (Both squares and segmentation algorithms)
Look at Saturation, high sat values might correlate to targets

//Edge detection---------------------------
See if Canny detection + Hough transform and a test for length of the resulting lines
Works well for the tan wooden beams and very thin wires that corner detection is having trouble with.

//Scoring Methods---------------------------
Run Waterfall and add up scores in each segment, use the highest?
Add up overall scores after filtering out outliers/single solo pixel (SSP) items
Run sequential detection methods
//-------------------------------------------
IMPORTANT TODO!!!!!!!!!!!!!!
EVERYTHING IS CAPSLOCK BECAUSE IT'S IMPORTANT!
NOTHING HERE RIGHT NOW!
//---------------------------------------------------------------------------
*/


#include "PrepareImage.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "tinydir.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

string IMG_EXT[] = { "jpg", "jpeg", "png", "bmp" };
string IMAGE_DIR = "";	//Stores Argv[1]

/// Global variables
Mat src, src_gray, crn, src2;
int total, positive, corner, edge = 0;
int total2, positive2, corner2, edge2 = 0;

//Debug control
int debug = 0;		//0 is off, 1 is on.  Don't make em anything else.
int dbDraw = 0;		//This should be common sense.  Don't add more values later.

//Parameters
int gBlurSize = 13;			//MUST BE ODD.
int cornerRange = 40;
int maxCorners = 25;
int edgePerFeature = 3;
int erosion_size = 2;
//Green Detection------------
int hueThreshLow = 30;
int hueThreshHigh = 80;
//Brown detection------------
int hueThresh2Low = 10;	
int hueThresh2High = 59;
int satThresh2Low = 50;
//Highlight Detection--------
int valThreshSpecHigh = 240;
int satThreshSpecLow = 10;
//Canny Parameters
int cannyEdgeThresh = 1;
int cannyLowThreshold = 30;
int cannyRatio = 3;
int cannyKernelSize = 3;
//Hough Parameters
int houghLineThresh = 1;
int houghUpperLimit = 30;
int houghSensitivity = 240; //Lower is more sensitive


//Aesthetic
int radii = 8;		//Radius for the corner markers

int maxTrackbar = 100;

RNG rng(12345);
char* source_window = "Image";
char* output_window = "Output";

/// Function header
void goodFeaturesToTrack_Demo(int, void*);

//stops all execution permanently, keeps window open.  For debug purposes.
void stop() {
	while (true)
		waitKey(0);
}

Mat CannyDetect(Mat src) {
	Mat dst, detectedEdges;
	dst.create(src.size(), src.type());
	cvtColor(src, src, CV_BGR2GRAY);
	blur(src, detectedEdges, Size(3, 3));
	Canny(detectedEdges, detectedEdges, cannyLowThreshold, cannyLowThreshold*cannyRatio, cannyKernelSize);
	dst = Scalar::all(0);
	src.copyTo(dst, detectedEdges);
	return dst;
}

int DistBet2Points(Point2f first, Point2f second) {
	return (sqrt(pow((first.x - second.x), 2) + (pow((first.y + - second.y), 2))));
}

bool SegmentExists(vector<pair<Point2f, Point2f>> lineVec, pair<Point2f, Point2f> line) {
	//Checks if a segment or it's opposite exists in the vector of line segments.
	for (int i = 0; i < lineVec.size(); i++) {
		if (lineVec[i] == line)
			return true;
		if ((lineVec[i].first == line.second) && (lineVec[i].second == line.first))
			return true;
	}
	return false;
}

vector<pair<Point2f, Point2f>> SegmentCorners(vector<Point2f> corners, Mat copy) {
	//Divides the vector of corners into a shape and plots it on the screen.
	vector<pair<Point2f, Point2f>> lineVec;
	pair<Point2f, Point2f> temp;
	int index = 0;
	int cur = 0;
	int min = INT_MAX;
	for (int q = 0; q < corners.size(); q++) {
		min = INT_MAX;
		for (cur = 0; cur < corners.size(); cur++) {
			if (cur != q) {
				temp = make_pair(corners[cur], corners[q]);
				if (DistBet2Points(corners[cur], corners[q]) <= cornerRange) {
					if (!SegmentExists(lineVec, temp)) {
						line(copy, corners[cur], corners[q], Scalar(0, 0, 255), 2, 8, 0);
						lineVec.push_back(temp);
						if (debug == 1)
							cout << "Line Segment: " << corners[cur].x << ", " << corners[cur].y
							<< ";	" << corners[q].x << ", " << corners[q].y << ".  They are " << DistBet2Points(corners[cur], corners[q])
							<< " far apart." << "\n";
					}
				}
			}
		}
		//After inner for		
	}
	//After outer for
	return lineVec;
}

Mat segmentImage(Mat src, vector<Point2i> markers) {
	//Watershed transform.  Not implemented or used yet.
	//STUB
	Mat wsh;
	cvtColor(src, wsh, CV_BGR2GRAY);
	if (!markers.size() == 0) {
		//Implement markers
		return wsh;
	}
	else {


		return wsh;
	}
}

//Removes Shi-Tomasi corners based on their surrounding HSV values.  This function removes points solely surrounded by green.  Parameters at the top.
vector<Point2f> RemoveColored(vector<Point2f> corners, int hueLower, int hueUpper, Mat copy) {
	cout << "** Removing Corners Based on Local HSV**" << endl;
	bool keep = false;
	int hue, sat, val;
	for (int i = 0; i < corners.size(); i++) {
		keep = false;
		for (int q = -1; q <= 1; q += 1) {		//x offset
			for (int w = -1; w <= 1; w += 1) {	//y offset
				hue = copy.at<cv::Vec3b>(corners[i].y + q, corners[i].x + w)[0];
				sat = copy.at<cv::Vec3b>(corners[i].y + q, corners[i].x + w)[1];
				val = copy.at<cv::Vec3b>(corners[i].y + q, corners[i].x + w)[2];
				if (!(((hueLower <= hue) && (hue <= hueUpper))   )) 				//if the color around the corner falls within the range of the two scalars
					keep = true;
			}
		}

		if (keep == false) {
			if (dbDraw == 1)
				circle(copy, corners[i], radii, Scalar(60, 255, 200), -1, 8, 0);	//Draws removed corners in green.
			corners[i] = Point2f(0, 0);
		}
	}
	
	corners.erase(std::remove_if(corners.begin(), corners.end(), [](Point2f i){ return i == Point2f(0, 0); }), corners.end());
	return corners;
}

//Removes Shi-Tomasi corners based on their surrounding HSV values.  This function removes points solely surrounded by brown/tan.  Parameters at the top.
vector<Point2f> RemoveBrown(vector<Point2f> corners, Mat copy) {
	cout << "** Removing Corners Based on Local HSV (Brown)**" << endl;
	bool keep = false;
	int hue, sat, val;
	for (int i = 0; i < corners.size(); i++) {
		keep = false;
		for (int q = -1; q <= 1; q += 1) {		//x offset
			for (int w = -1; w <= 1; w += 1) {	//y offset
				hue = copy.at<cv::Vec3b>(corners[i].y + q, corners[i].x + w)[0];
				sat = copy.at<cv::Vec3b>(corners[i].y + q, corners[i].x + w)[1];
				val = copy.at<cv::Vec3b>(corners[i].y + q, corners[i].x + w)[2];
				if (!(((hueThresh2Low <= hue) && (hue <= hueThresh2High)) && sat >= satThresh2Low)) 				//if the color around the corner falls within the range of the two scalars
					keep = true;
			}
		}				//Add value and saturation threshold.  High val or low sat should keep

		if (keep == false) {
			if (dbDraw == 1)
				circle(copy, corners[i], radii, Scalar(30, 255, 200), -1, 8, 0);	//Draws removed corners in yellow.
			corners[i] = Point2f(0, 0);
		}
	}

	corners.erase(std::remove_if(corners.begin(), corners.end(), [](Point2f i){ return i == Point2f(0, 0); }), corners.end());
	return corners;
}

//Removes Shi-Tomasi corners based on their surrounding HSV values.  This function removes points with extremely high value/low saturation.  Parameters at the top.
vector<Point2f> RemoveSpecular(vector<Point2f> corners, Mat copy) {
	cout << "** Removing Corners Based on Local HSV (Specular)**" << endl;
	bool keep = false;
	int hue, sat, val;
	for (int i = 0; i < corners.size(); i++) {
		keep = false;
		for (int q = -1; q <= 1; q += 1) {		//x offset
			for (int w = -1; w <= 1; w += 1) {	//y offset
				hue = copy.at<cv::Vec3b>(corners[i].y + q, corners[i].x + w)[0];
				sat = copy.at<cv::Vec3b>(corners[i].y + q, corners[i].x + w)[1];
				val = copy.at<cv::Vec3b>(corners[i].y + q, corners[i].x + w)[2];
				if (( sat >= satThreshSpecLow) && (val <= valThreshSpecHigh)) 				
					keep = true;
			}
		}	

		if (keep == false) {
			if (debug == 1)
				cout << "Removing a corner based on local HSV at: " << corners[i].x << ", " << corners[i].y << "\n";
			if (dbDraw == 1)
				circle(copy, corners[i], radii, Scalar(0, 0, 0), -1, 8, 0);	//Draws removed corners in black.
			corners[i] = Point2f(0, 0);
		}
	}

	corners.erase(std::remove_if(corners.begin(), corners.end(), [](Point2f i){ return i == Point2f(0, 0); }), corners.end());
	return corners;
}

bool WithinRange(Point2f first, Point2f second, int range) {
	if (range > sqrt(pow((first.x - second.x), 2) + (pow((first.y + - second.y), 2)))) {
		//cout << sqrt(pow((first.x - second.x), 2) + (pow((first.y + -second.y), 2)));
		return true;
	}
	else
		return false;
}

//Removes isolated points
vector<Point2f> RemoveSolo(vector<Point2f> corners, Mat copy) {
	cout << "** Removing Isolated Corners **" << endl;
	bool keep = false;
	for (int i = 0; i < corners.size(); i++) {
		keep = false;
		for (int j = 0; j < corners.size(); j++) {
			if (i != j) {
				if (true == WithinRange(corners[j], corners[i], cornerRange)) {
					keep = true;
				}
				//If There is another corner within range
			}
		}
		if (keep == false) {
			if (dbDraw == 1)
				circle(copy, corners[i], radii, Scalar(255, 0, 0), -1, 8, 0);	//Draws removed corners in blue.
			corners[i] = Point2f(0,0);
		}
	}
	corners.erase(std::remove_if(corners.begin(), corners.end(), [](Point2f i){ return i == Point2f(0, 0); }), corners.end());
	return corners;
}

int InBox(vector<vector<pair<Point2f, Point2f>>> vec, pair<Point2f, Point2f> target) {
	//Checks whether or not either endpoint of a given line is in a set of line segments.

	//Deprecated debug.  Unless you're modifying this, you don't need these messages.
	//if (debug == 1)
	//	cout << "Target Segment: " << target.first.x << ", " << target.first.y
	//	<< ";	" << target.second.x << ", " << target.second.y << ".\n";
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[i].size(); j++) {
			if ((vec[i][j].second == target.first) || (vec[i][j].first == target.second) || (vec[i][j].first == target.first) || (vec[i][j].second == target.second))
				return i;
		}
	}
	return -1;
}

int BothInBox(vector<vector<pair<Point2f, Point2f>>> vec, pair<Point2f, Point2f> target) {
	//Unlike inbox, both points must be in the vector for it to return true.
	//Checks whether or not a given line is in a set of line segments.

	//Deprecated debug.  Unless you're modifying this, you don't need these messages.
	//if (debug == 1)
	//	cout << "Target Segment: " << target.first.x << ", " << target.first.y
	//	<< ";	" << target.second.x << ", " << target.second.y << ".\n";
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[i].size(); j++) {
			if (((vec[i][j].first == target.first) && (vec[i][j].second == target.first)) || ((vec[i][j].first == target.first) && (vec[i][j].second == target.second)))
				return i;
		}
	}
	return -1;
}

bool ScoreImage(vector<pair<Point2f, Point2f>> lineVect) {
	//Returns true if the image is judged to contain debris, false otherwise.
	bool containsDebris = false;
	pair<Point2f, Point2f> temp;
	int biggestBox = 0;
	int i = 0;
	bool exists = false;			//Used to keep track of whether or not a 2nd segment is already in a box.
	vector<pair<Point2f, Point2f>> tempVec(1);
	vector<vector<pair<Point2f, Point2f>>> boxVec;
	//Simple version, if we have more than edgePerFeature connected points, it's got debris.
	for (int self = 0; self < lineVect.size(); self++) {
		//put self in a box
		for (int other = 0; other < lineVect.size(); other++) {
			if ((lineVect[self].second == lineVect[other].first) || (lineVect[self].first == lineVect[other].second) || (lineVect[self].first == lineVect[other].first)) {
				temp = lineVect[self];
				i = InBox(boxVec, temp);
				//if (debug == 1)
				//	cout << "Point " << lineVect[self].first << ", " << lineVect[self].second << " is in box: " << i << endl;
				if (i == -1) {
					//If -1, means there is no box for the current line.
					tempVec[0] = lineVect[self];
					boxVec.push_back(tempVec);
					if (tempVec[0] != lineVect[other])
						boxVec[boxVec.size() - 1].push_back(lineVect[other]);
					//Put self and other in a new box
				}
				else {
					//Check if the 2nd segment is in an existing box.
					if (BothInBox(boxVec, temp) != i)
						boxVec[i].push_back(temp);
				}
			}
		}
		if (debug == 2) {
			cout << "Iteration: " << self << "\n";
			for (int g = 0; g < boxVec.size(); g++) {
				cout << "Box number: " << g << " contains:\n";
				for (int h = 0; h < boxVec[g].size(); h++) {
					cout << "Segment " << h << " contains " << boxVec[g][h].first << ", " << boxVec[g][h].second << endl;
					cout << "";
				}
			}
			cout << "end of iteration recap\n";
		}
	}
	
	for (int a = 0; a < boxVec.size(); a++) {
		if (boxVec[a].size() > biggestBox)
			biggestBox = boxVec[a].size();
	}
	cout << "Largest feature contains " << biggestBox << " lines.\n";
	if (debug == 1) {
		cout << "Box sizes: \n";
		for (int a = 0; a < boxVec.size(); a++) {
			cout << boxVec[a].size() << endl;
		}
	}

	if (biggestBox >= edgePerFeature)
		containsDebris = true;
	return containsDebris;
}

vector<string> readDir(string dir_name)
{
	tinydir_dir dir;
	vector<string> files;
	if (tinydir_open(&dir, dir_name.c_str()) == -1) {
		cerr << "Couldn't find directory\n";
		strerror(errno);
	}
	//cerr << "Entering While next loop\n";
	while (dir.has_next)
	{
		tinydir_file file;
		//cerr << "Reading file\n";
		tinydir_readfile(&dir, &file);
		//cerr << "Read file\n";
		string ext = string(file.extension);
		transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

		if (!file.is_dir && find(begin(IMG_EXT), end(IMG_EXT), ext) != end(IMG_EXT))
		{
			files.push_back(file.name);
			//cout << file.name << endl;
		}

		tinydir_next(&dir);
	}

	tinydir_close(&dir);
	return files;
}

Mat RemoveGreenPixels(Mat src) {
	int hue, sat, val;
	cvtColor(src, src, CV_BGR2HSV);
	for (int x = 0; x < src.cols; ++x) {
		for (int y = 0; y < src.rows; ++y) {
			hue = src.at<cv::Vec3b>(y, x)[0];
			sat = src.at<cv::Vec3b>(y, x)[1];
			val = src.at<cv::Vec3b>(y, x)[2];

			if ((hue >= hueThreshLow) && (hue <= hueThreshHigh)) {
				src.at<cv::Vec3b>(y, x)[0] = 0;
				src.at<cv::Vec3b>(y, x)[1] = 0;
				src.at<cv::Vec3b>(y, x)[2] = 0;
			}
		}
	}
	cvtColor(src, src, CV_HSV2BGR);
	return src;
}

Mat RemoveBrownPixels(Mat src) {
	int hue, sat, val;
	cvtColor(src, src, CV_BGR2HSV);
	for (int x = 0; x < src.cols; ++x) {
		for (int y = 0; y < src.rows; ++y) {
			hue = src.at<cv::Vec3b>(y, x)[0];
			sat = src.at<cv::Vec3b>(y, x)[1];
			val = src.at<cv::Vec3b>(y, x)[2];

			if ((hue >= hueThresh2Low) && (hue <= hueThreshHigh) && sat >= satThresh2Low) {
				src.at<cv::Vec3b>(y, x)[0] = 0;
				src.at<cv::Vec3b>(y, x)[1] = 0;
				src.at<cv::Vec3b>(y, x)[2] = 0;
			}
		}
	}
	cvtColor(src, src, CV_HSV2BGR);
	return src;
}

bool BatchDetectCorners(Mat src) {
	//Runs full preprocessing and S-T detection on an image, then returns the result
	bool result = false;
	int erosion_size = 1;
	Mat element = getStructuringElement(cv::MORPH_CROSS,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	erode(src, crn, element);
	erode(crn, crn, element);
	GaussianBlur(crn, crn, Size(gBlurSize, gBlurSize), 0, 0 );
	crn = bilateralBlur(crn);		//Resizes to 1/4 size too

	cv::cvtColor(crn, src_gray, CV_BGR2GRAY);
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	Mat copy;
	Mat hsv;
	copy = crn.clone();
	cvtColor(copy, copy, CV_BGR2HSV);
	goodFeaturesToTrack(src_gray,
		corners,
		maxCorners,
		qualityLevel,
		minDistance,
		Mat(),
		blockSize,
		useHarrisDetector,		//( = False)
		k);

	vector<Point2f>corners2 = RemoveColored(corners, hueThreshLow, hueThreshHigh, copy);
	corners2 = RemoveBrown(corners2, copy);
	corners2 = RemoveSpecular(corners2, copy);
	cvtColor(copy, copy, CV_HSV2BGR);
	corners2 = RemoveSolo(corners2, copy);
	//for (int i = 0; i < corners2.size(); i++) {
	//	circle(copy, corners2[i], radii, Scalar(rng.uniform(0, 0), rng.uniform(0, 10), rng.uniform(255, 255)), -1, 8, 0);
	//}
	vector<pair<Point2f, Point2f>> lineVect = SegmentCorners(corners2, copy);
	result = ScoreImage(lineVect);
	if (dbDraw == 1) {
		cv::imshow(output_window, copy);
		cv::waitKey(1);
	}
	return result;
}

bool BatchDetectEdges(Mat src) {
	//Preprocesses and runs canny edge detection + Hough transform.
	//If lines exist, it is marked as containing debris.  Only grabs long lines.
	//This method is intended to complement the corner check by grabbing features it misses with very minimal false postives
	Mat element = getStructuringElement(cv::MORPH_CROSS,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));
	dilate(src, src, element);
	dilate(src, src, element);
	GaussianBlur(src, src, Size(gBlurSize, gBlurSize), 0, 0);

	src = bilateralBlur(src);		//As implemented, does resize to size/4

	src = RemoveGreenPixels(src);
	//src = RemoveBrownPixels(src);

	Mat canny, cdst;
	//Canny detection, parameters are global.
	canny = CannyDetect(src);
	//Hough Transform.  Both methods included.  These have local parameters.
	//Parameters for the batch version are up top by the corner parameters.
	cv::cvtColor(canny, cdst, CV_GRAY2BGR);
	vector<Vec2f> lines;
	HoughLines(canny, lines, 1, CV_PI / 180, houghSensitivity, 0, 0);
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
		}
	if (dbDraw == 1) {
		cv::imshow(output_window, cdst);
		cv::imshow(source_window, src);
	}
	waitKey(1);
	if (debug == 1) {
		cout << "Number of edges detected: " << lines.size() << endl;
	}
	if ((lines.size() >= houghLineThresh) && (lines.size() <= houghUpperLimit))
		return true;
}

int main(int argc, char** argv)
{
	std::cout << std::boolalpha;
	int mode = 5;
	//src2 = imread(argv[1], 1);

	if (debug == 1)
		cout << "0 = Shi-Tom, 1 = Color Detection, 2 = Erode, 3 = Show Single Channel, 5 = Batch Checker!=, 7 = run general test\n";

	string modeStr;		
	if (debug == 1) {
		cin >> modeStr;
		mode = strtod(modeStr.c_str(), nullptr);
	}
	if (dbDraw == 1) {
		cv::namedWindow(source_window, CV_WINDOW_NORMAL);
		cv::namedWindow(output_window, CV_WINDOW_NORMAL);
	}

	if (mode == 0) {
		cout << "Mode 0\n"; 
		src = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

		if (!src.data)                              // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		Mat element = getStructuringElement(cv::MORPH_CROSS,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		
		erode(src, crn, element);
		crn = bilateralBlur(crn);		//As implemented, does resize to size/4
		
		cv::cvtColor(crn, src_gray, CV_BGR2GRAY);

		/// Create Trackbar to set the number of corners
		cv::createTrackbar("Max  corners:", output_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo);

		cv::imshow(source_window, src);

		goodFeaturesToTrack_Demo(0, 0);
	}

	if (mode == 1) {
		cout << "Mode 1\n";
		// Convert the image into an HSV image
		src = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

		if (!src.data) {
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		resize(src, src, Size(int(src.cols / 4), int(src.rows / 4)));


		Mat hsv;
		cvtColor(src, hsv, CV_BGR2HSV);

		//Hue of 24-50 is brown, 75 to 135 is green/greenish, saturation < 25 is whiteish, brightness + saturation < 40 is gray or black
		//All hue values get divided by 2 in opencv.

		//if (hue is 12 to 25 AND sat is less than 70% (178))
			//It's brown/tan, ignore it.

		//if (hue is 37 to 67 AND bright is less than 70)
			//It's green/vegetation.  High bright green could be abnormal.

		//Attempt to threshold out everything normal, leave only weird colors.
		int hue, sat, val;

		for (int x = 0; x < hsv.cols; ++x) {
			for (int y = 0; y < hsv.rows; ++y) {
				hue = hsv.at<cv::Vec3b>(y, x)[0];
				sat = hsv.at<cv::Vec3b>(y, x)[1];
				val = hsv.at<cv::Vec3b>(y, x)[2];
				
				if ((sat >= satThreshSpecLow) && (val <= valThreshSpecHigh)) {
					hsv.at<cv::Vec3b>(y, x)[0] = 0;
					hsv.at<cv::Vec3b>(y, x)[1] = 0;
					hsv.at<cv::Vec3b>(y, x)[2] = 0;
				}
			}
		}
		cv::imshow(output_window, hsv);
		cv::imshow(source_window, src);
	}

	if (mode == 2) {
		cout << "Mode 2\n";
		//Do Morphology operations to get rid of small bright spots, generally specular reflections over water.
		src = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

		if (!src.data)                              // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		Mat dst;
		resize(src, src, Size(int(src.cols / 4), int(src.rows / 4)));
		//erode(src, dst, Mat(), Point(-1, -1), 2, 1, 1);
		erosion_size = 6;
		Mat element = getStructuringElement(cv::MORPH_CROSS,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		erode(src, dst, element);

		cv::imshow(output_window, dst);
		cv::imshow(source_window, src);
	}

	if (mode == 3) {
		cout << "Mode 3\n";
		//Another attempt to get rid of specular reflections over water.
		//Threshold out anything with high brightness and low saturation.
		src = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

		if (!src.data)                              // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		Mat hsv;
		resize(src, src, Size(int(src.cols / 4), int(src.rows / 4)));
		src = hsv.clone();
		cvtColor(hsv, hsv, CV_BGR2HSV);

		int hue, sat, val;

		for (int x = 0; x < hsv.cols; ++x) {
			for (int y = 0; y < hsv.rows; ++y) {
				hue = hsv.at<cv::Vec3b>(y, x)[0];
				sat = hsv.at<cv::Vec3b>(y, x)[1];
				val = hsv.at<cv::Vec3b>(y, x)[2];

				//if ((sat < 40) && (val > 150) ) {
				//hsv.at<cv::Vec3b>(y, x)[0] = 0;
				hsv.at<cv::Vec3b>(y, x)[1] = 0;
				hsv.at<cv::Vec3b>(y, x)[2] = 0;
				//}
			}
		}

		cv::imshow(output_window, hsv);
		cv::imshow(source_window, src);
	}
	if (mode == 5) {
		cout << "Mode 5: Batch Processing\n";
		IMAGE_DIR = argv[1];
		vector<string> files = readDir(IMAGE_DIR);
		string targDir = IMAGE_DIR + "/HasDebris/";
		CreateDirectory(targDir.c_str(), NULL);
		int total = 0;
		int positive = 0;
		int corner = 0;
		int edge = 0;
		bool cur = false;
		for (int i = 0; i < files.size(); i++) {
			//Judge the pictures
			src = imread((IMAGE_DIR + '/' + files[i]), 1); 
			if (debug == 1)
				cout << endl << "Image:" << files[i] << endl;
			cur = BatchDetectEdges(src);
			if (cur == true) {
				positive++;
				edge++;
			}
			else {
				cur = BatchDetectCorners(src);
				if (cur == true) {
					positive++;
					corner++;
				}
			}
			if (cur == true) {
				//save the file
				imwrite((targDir + files[i]), src);
				cout << (IMAGE_DIR + "/" + files[i])  << "\n";
				//remove((IMAGE_DIR + "/" + files[i]).c_str());
			}
			total++;
			cout << "Completed: " << total << endl;
			//cout << "Edge: " << edge << endl;
			//cout << "Corner: " << corner << endl;
			//cout << "Correct: " << positive << endl;
		}


	}
	if (mode == 6) {
		cout << "Mode - Edge Detection + Hough transform.\n";
		src = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

		if (!src.data)                              // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		//resize(src, src, Size(int(src.cols / 4), int(src.rows / 4)));
		Mat element = getStructuringElement(cv::MORPH_CROSS,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		dilate(src, src, element);
		GaussianBlur(src, src, Size(gBlurSize, gBlurSize), 0, 0);

		src = bilateralBlur(src);		//As implemented, does resize to size/4
		Mat canny, cdst;
		//Canny detection, parameters are global.
		canny = CannyDetect(src);
		//Hough Transform.  Both methods included.  These have local parameters.
		//Parameters for the batch version are up top by the corner parameters.
		cvtColor(canny, cdst, CV_GRAY2BGR);
		bool cannyp = false;
		bool cannyb = true;
		if (cannyb) {
			vector<Vec2f> lines;
			HoughLines(canny, lines, 1, CV_PI / 180, 250, 0, 0);
			for (size_t i = 0; i < lines.size(); i++)
			{
				float rho = lines[i][0], theta = lines[i][1];
				Point pt1, pt2;
				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;
				pt1.x = cvRound(x0 + 1000 * (-b));
				pt1.y = cvRound(y0 + 1000 * (a));
				pt2.x = cvRound(x0 - 1000 * (-b));
				pt2.y = cvRound(y0 - 1000 * (a));
				line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
			}
			cv::imshow(output_window, cdst);
			cout << lines.size();
		}
		if (cannyp) {
			vector<Vec4i> lines;
			HoughLinesP(canny, lines, 1, CV_PI / 180, 50, 50, 10);
			for (size_t i = 0; i < lines.size(); i++)
			{
				Vec4i l = lines[i];
				line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
			}


			cv::imshow(output_window, cdst);
			cout << lines.size();
		}

		if (!(cannyb || cannyp))
			cv::imshow(output_window, canny);
		cv::imshow(source_window, src);
	}
	if (mode == 7) {
		cout << "Mode 7: Full Test\n";
		IMAGE_DIR = argv[1];
		for (int q = 0; q < 2; q++) {
			vector<string> files;

			if (q == 0)
				files = readDir(IMAGE_DIR + '/' + "Debris");
			else
				files = readDir(IMAGE_DIR + '/' + "NoDebris");
			bool cur = false;
			for (int i = 0; i < files.size(); i++) {
				//Judge the pictures
				if (q == 0)
				src = imread((IMAGE_DIR + "/Debris/" + files[i]), 1);
				else
				src = imread((IMAGE_DIR + "/NoDebris/" + files[i]), 1);
				if (!src.data) 	{ return -1; }
				if (debug == 1)
					cout << endl << "Image:" << files[i] << endl;
				if (q == 0) {
					cur = BatchDetectEdges(src);
					if (cur == true) {
						positive++;
						edge++;
					}
					else {
						cur = BatchDetectCorners(src);
						if (cur == true) {
							positive++;
							corner++;
						}
					}
					total++;
					cout << "Completed: " << total << endl;
					cout << "Edge: " << edge << endl;
					cout << "Corner: " << corner << endl;
					cout << "Correct: " << positive << endl;
				}
				else {
					cur = BatchDetectEdges(src);
					if (cur == true) {
						positive2++;
						edge2++;
					}
					else {
						cur = BatchDetectCorners(src);
						if (cur == true) {
							positive2++;
							corner2++;
						}
					}
					total2++;
					cout << "Completed: " << total2 << endl;
					cout << "Edge: " << edge2 << endl;
					cout << "Corner: " << corner2 << endl;
					cout << "Correct: " << positive2 << endl;
				}

			}
		}
			//Below for loop
			double percentage, percentage2;
			percentage = (positive * 100 / total);
			percentage2 = (positive2 * 100 / total2);
			cout << "Percent correct: " << percentage << "%\n";
			cout << "Debris Pictures: : " << total << " images" << endl;
			cout << "Marked for Edges: " << edge << endl;
			cout << "Marked for Corners: " << corner << endl;
			cout << "Correct: " << positive << endl;
			cout << "Non-Debris Pictures: " << total2 << " images" << endl;
			cout << "Marked for Edges: " << edge2 << endl;
			cout << "Marked for Corners: " << corner2 << endl;
			cout << "Total false positives: " << positive2 << endl;
			cout << "Percent false positive: " << percentage2 << "%\n";
	}
	
	cv::waitKey(0);
	return(0);
}


//Apply Shi-Tomasi corner detector
void goodFeaturesToTrack_Demo(int, void*)
{
	if (maxCorners < 1) { maxCorners = 1; }

	/// Parameters for Shi-Tomasi algorithm
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;

	/// Copy the filtered source image
	Mat copy;
	copy = crn.clone();

	/// Apply corner detection
	goodFeaturesToTrack(src_gray,
		corners,
		maxCorners,
		qualityLevel,
		minDistance,
		Mat(),
		blockSize,
		useHarrisDetector,		//( = False)
		k);



	/// Draw corners detected
	//Largely deprecated, doesn't currently use all filters.  Just need to add remove brown and specular calls. See batch module
	if (debug == 1)
		cout << "** Number of corners detected: " << corners.size() << endl;
	vector<Point2f>corners2 = RemoveColored(corners, hueThreshLow, hueThreshHigh, copy);
	if (debug == 1)
		cout << "** Number of corners detected: " << corners2.size() << endl;
	corners2 = RemoveSolo(corners2, copy);
	if (debug == 1)
		cout << "** Number of corners detected: " << corners2.size() << endl;
	for (int i = 0; i < corners2.size(); i++) {
		circle(copy, corners2[i], radii, Scalar(rng.uniform(0, 0), rng.uniform(0, 10), rng.uniform(255, 255)), -1, 8, 0);
	}
	vector<pair<Point2f, Point2f>> lineVect = SegmentCorners(corners2, copy);
	circle(copy, Point2i(1385,645), 100, Scalar(rng.uniform(0, 0), rng.uniform(0, 10), rng.uniform(255, 255)), 4, 8, 0);
	cout  << ScoreImage(lineVect);

	cv::imshow(output_window, copy);
}
