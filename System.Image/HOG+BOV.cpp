#include "../System/System.h"
#include "Feature.h"
#include "BOV.h"
#include <cv.h>
#include <highgui.h>
#include <tuple>
using namespace System::IO;
using namespace System::Image;
using namespace std;
using namespace cv;

int main()
{
    DirectoryInfo dataSetInfo("oracles_png");

	vector<System::String> classInfos = dataSetInfo.GetDirectories();
	sort(classInfos.begin(), classInfos.end());

	vector<tuple<System::String, int>> files;
	for (int i = 0; i < classInfos.size(); i++)
	{
		vector<System::String> fileInfos = DirectoryInfo(classInfos[i]).GetFiles();
		sort(fileInfos.begin(), fileInfos.end());
		
		for (int j = 0; j < fileInfos.size(); j++)
			files.push_back(make_tuple(fileInfos[j], i + 1));
	}

	vector<Mat> images;
	for (int i = 0; i < files.size(); i++)
		images.push_back(imread(get<0>(files[i]), CV_LOAD_IMAGE_GRAYSCALE));

	int imageNum = images.size();
	vector<Feature> features(imageNum);
	printf("Compute Features...\n");
	#pragma omp parallel for
	for (int i = 0; i < imageNum; i++)
	{
		HOG hog;
		features[i] = hog.GetFeature(images[i], true);
	}

	int fold = 3;
	vector<Feature> trainingFeatures;
	vector<int> randomIndexes = RandomPickUp(imageNum, imageNum / fold);
	int counter = 0;
	for (int i = 0; i < imageNum; i++)
	{
		if (counter < randomIndexes.size() && randomIndexes[counter] != i)
			trainingFeatures.push_back(features[i]);
		else
			counter++;
	}

	vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 500, 1000000);
	vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

	printf("Writing To File...\n");
	FILE* file = fopen("training", "w");
	for (int i = 0; i < freqHistograms.size(); i++)
	{
		fprintf(file, "%d", get<1>(files[i]));
		for (int j = 0; j < freqHistograms[i].size(); j++)
			fprintf(file, " %d:%f", j + 1, freqHistograms[i][j]);
		fprintf(file, "\n");
	}
	fclose(file);
}