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

int main()
{
	//BHOG("oracles_png", "bhog_oracles_data");

	Mat image = imread("00001.png", CV_LOAD_IMAGE_GRAYSCALE);

	Mat revImage = reverse(image), thre, thinned;
	threshold(revImage, thre, 127, 1, CV_THRESH_BINARY);
	thin(thre, thinned);

	tuple<vector<System::Image::Point>, vector<System::Image::Point>> 
		points = FindJunctionsOrEndPoints(thinned); 
	vector<System::Image::Point>& junc = get<0>(points);
	vector<System::Image::Point>& endP = get<1>(points);

	Mat colorImage;
	threshold(thinned, thre, 0.5, 255, CV_THRESH_BINARY);
	cvtColor(thre, colorImage, CV_GRAY2BGR);
	for (int i = 0; i < junc.size(); i++)
		circle(colorImage, cv::Point(get<1>(junc[i]), get<0>(junc[i])), 5, Scalar(255, 0, 0, 255));

	imshow("win", colorImage);
	waitKey(0);
}