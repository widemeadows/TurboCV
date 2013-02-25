#pragma once

#include "Util.h"
#include "Feature.h"
#include <cv.h>
using namespace std;

namespace System
{
	namespace Image
	{
		class BOV
		{
		public:
			static vector<Descriptor> GetVisualWords(const vector<Feature>& features, 
				int clusterNum, int sampleNum = INF);

			static vector<Histogram> BOV::GetFrequencyHistograms(const vector<Feature>& features, 
				const vector<Descriptor>& words);

			static vector<Histogram> ComputeFrequencyHistogram(const vector<Feature>& features, 
				int clusterNum, int sampleNum = INF);
		
		private:
			static vector<double> GetDistancesToVisualWords(const Descriptor& descriptor, 
				const vector<Descriptor>& words);

			static Histogram BOV::GetFrequencyHistogram(const Feature& feature, 
				const vector<Descriptor>& words);
		};

		inline vector<Descriptor> BOV::GetVisualWords(const vector<Feature>& features, 
			int clusterNum, int sampleNum)
		{
			vector<Descriptor> allDescriptors;
			for (int i = 0; i < features.size(); i++)
				for (int j = 0; j < features[i].size(); j++)
					allDescriptors.push_back(features[i][j]);

			assert(allDescriptors.size() > 0);
			int descriptorNum = allDescriptors.size(),
				descriptorSize = allDescriptors[0].size();
			printf("Descriptor Num: %d, Descriptor Size: %d.\n", 
				descriptorNum, descriptorSize);

			sampleNum = min((int)allDescriptors.size(), sampleNum);
			Mat samples(sampleNum, descriptorSize, CV_32F);
			vector<int> randomIndex = RandomPickUp(descriptorNum, sampleNum);

			int counter = 0;
			for (int i = 0; i < randomIndex.size(); i++)
			{
				int index = randomIndex[i];

				for (int j = 0; j < descriptorSize; j++)
					samples.at<float>(counter, j) = allDescriptors[index][j];

				counter++;
			}

			Mat labels(sampleNum, 1, CV_32S);
			for (int i = 0; i < sampleNum; i++)
				labels.at<int>(i, 0) = 0;

			Mat centres(clusterNum, descriptorSize, CV_32F);
			printf("K-Means Begin...\n");
			kmeans(samples, clusterNum, labels, TermCriteria(CV_TERMCRIT_ITER, 500, 1e-6), 
				1, KMEANS_PP_CENTERS, centres);

			vector<Descriptor> words(clusterNum);
			for (int i = 0; i < clusterNum; i++)
				for (int j = 0; j < descriptorSize; j++)
					words[i].push_back(centres.at<float>(i, j));

			return words;
		}

		inline vector<double> BOV::GetDistancesToVisualWords(const Descriptor& descriptor, 
			const vector<Descriptor>& words)
		{
			assert(words.size() > 0 && descriptor.size() == words[0].size());

			int wordNum = words.size();
			vector<double> distances;

			for (int i = 0; i < wordNum; i++)
				distances.push_back(GaussianDistance(descriptor, words[i], 0.1));

			NormOneNormalize(distances);
			return distances;
		}

		inline Histogram BOV::GetFrequencyHistogram(const Feature& feature, 
			const vector<Descriptor>& words)
		{	
			int wordNum = words.size();
			Histogram freqHistogram(wordNum, 0);

			for (int i = 0; i < feature.size(); i++)
			{
				vector<double> distances = GetDistancesToVisualWords(feature[i], words);
				for (int j = wordNum - 1; j >= 0; j--)
					freqHistogram[j] += distances[j];
			}

			return freqHistogram;
		}

		inline vector<Histogram> BOV::GetFrequencyHistograms(const vector<Feature>& features, 
			const vector<Descriptor>& words)
		{
			int imageNum = features.size();
			vector<Histogram> freqHistograms(imageNum);

			printf("Compute Frequency Histograms...\n");
			#pragma omp parallel for
			for (int i = 0; i < imageNum; i++)
				freqHistograms[i] = GetFrequencyHistogram(features[i], words);

			return freqHistograms;
		}

		inline vector<Histogram> BOV::ComputeFrequencyHistogram(const vector<Feature>& features,
			int clusterNum, int sampleNum)
		{
			return GetFrequencyHistograms(features, GetVisualWords(features, clusterNum, sampleNum));
		}
	}
}