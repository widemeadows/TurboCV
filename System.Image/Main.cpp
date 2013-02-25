#include "../System/System.h"
#include "System.Image.h"
#include "Algorithm.h"
using namespace System::Image;

void BHOG(const System::String& imageSetPath, const System::String& savePath)
{
	vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
	int imageNum = images.size();

	vector<Feature> features(imageNum);
	printf("Compute Features...\n");
	#pragma omp parallel for
	for (int i = 0; i < imageNum; i++)
		features[i] = HOG().GetFeature(get<0>(images[i]), true);
	
	vector<Feature> trainingFeatures = RandomPickUp(features, features.size() / 3);
	vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 500, 1000000);
	vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

	printf("Writing To File...\n");
	FILE* file = fopen(savePath, "w");
	for (int i = 0; i < freqHistograms.size(); i++)
	{
		fprintf(file, "%d", get<1>(images[i]));
		for (int j = 0; j < freqHistograms[i].size(); j++)
			fprintf(file, " %d:%f", j + 1, freqHistograms[i][j]);
		fprintf(file, "\n");
	}
	fclose(file);
}

Mat GetBoundingBox(const Mat& sketchImage)
{
    int minX = sketchImage.cols - 1, maxX = 0,
		minY = sketchImage.rows - 1, maxY = 0;

	for (int i = 0; i < sketchImage.rows; i++)
		for (int j = 0; j < sketchImage.cols; j++)
		{
			if (sketchImage.at<uchar>(i, j))
			{
				minX = std::min(minX, j);
				maxX = max(maxX, j);
				minY = std::min(minY, i);
				maxY = max(maxY, i);
			}
		}

	return Mat(sketchImage, Range(minY, maxY + 1), Range(minX, maxX + 1));
}

int main()
{
	//BHOG("oracles_png", "bhog_oracles_data");

	Mat image = imread("00001.png", CV_LOAD_IMAGE_GRAYSCALE);

    Mat revImage = reverse(image);

    Mat boundingBox = GetBoundingBox(revImage);

    Mat squareImage;
	int widthPadding = 0, heightPadding = 0;
	if (boundingBox.rows < boundingBox.cols)
		heightPadding = (boundingBox.cols - boundingBox.rows) / 2;
	else
		widthPadding = (boundingBox.rows - boundingBox.cols) / 2;
	copyMakeBorder(boundingBox, squareImage, heightPadding, heightPadding, 
        widthPadding, widthPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

    Mat scaledImage;
	resize(squareImage, scaledImage, Size(112, 112));

	Mat paddedImage;
	copyMakeBorder(scaledImage, paddedImage, 8, 8, 8, 8, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

	Mat binaryImage, thinnedImage;
    threshold(paddedImage, binaryImage, 54, 1, CV_THRESH_BINARY);
    thin(binaryImage, thinnedImage);

	tuple<vector<Point>, vector<Point>>	points = FindJunctionsOrEndpoints(thinnedImage); 
	vector<Point>& junc = get<0>(points);
	vector<Point>& endP = get<1>(points);

	Mat colorImage;
	threshold(thinnedImage, binaryImage, 0.5, 255, CV_THRESH_BINARY);
	cvtColor(binaryImage, colorImage, CV_GRAY2BGR);
	for (int i = 0; i < junc.size(); i++)
		circle(colorImage, junc[i], 5, Scalar(255, 0, 0, 255));
    for (int i = 0; i < endP.size(); i++)
		circle(colorImage, endP[i], 5, Scalar(0, 255, 0, 255));

	imshow("win", colorImage);
	waitKey(0);
}