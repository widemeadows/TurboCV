#pragma once

#include "../System/System.h"
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
            static vector<DescriptorInfo<float>> GetVisualWords(
                const vector<FeatureInfo<float>>& features, int clusterNum, int sampleNum = INF);

            static vector<Histogram> GetFrequencyHistograms(
                const vector<FeatureInfo<float>>& features, const vector<DescriptorInfo<float>>& words);

            static vector<Histogram> GetFrequencyHistogram(
                const vector<FeatureInfo<float>>& features, int clusterNum, int sampleNum = INF);
        
        private:
            static vector<double> GetDistancesToVisualWords(const DescriptorInfo<float>& descriptor, 
                const vector<DescriptorInfo<float>>& words);

            static Histogram BOV::GetFrequencyHistogram(const FeatureInfo<float>& feature, 
                const vector<DescriptorInfo<float>>& words);
        };

        inline vector<DescriptorInfo<float>> BOV::GetVisualWords(
            const vector<FeatureInfo<float>>& features, int clusterNum, int sampleNum)
        {
            vector<DescriptorInfo<float>> allDescriptors;
            for (int i = 0; i < features.size(); i++)
                for (int j = 0; j < features[i].size(); j++)
                    allDescriptors.push_back(features[i][j]);

            assert(allDescriptors.size() > 0);
            int descriptorNum = (int)allDescriptors.size(),
                descriptorSize = (int)allDescriptors[0].size();
            printf("Descriptor Num: %d, Descriptor Size: %d.\n", 
                descriptorNum, descriptorSize);

            sampleNum = min(descriptorNum, sampleNum);
            Mat samples(sampleNum, descriptorSize, CV_32F);
            vector<int> randomIndex = RandomPermutate(descriptorNum, sampleNum);
            sort(randomIndex.begin(), randomIndex.end());

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

            vector<DescriptorInfo<float>> words(clusterNum);
            for (int i = 0; i < clusterNum; i++)
                for (int j = 0; j < descriptorSize; j++)
                    words[i].push_back(centres.at<float>(i, j));

            return words;
        }

        inline vector<double> BOV::GetDistancesToVisualWords(
            const DescriptorInfo<float>& descriptor, const vector<DescriptorInfo<float>>& words)
        {
            assert(words.size() > 0 && descriptor.size() == words[0].size());

            int wordNum = (int)words.size();
            vector<double> distances;

            for (int i = 0; i < wordNum; i++)
                distances.push_back(Math::GaussianDistance(descriptor.getVec(), words[i].getVec(), 0.1));

            NormOneNormalize(distances.begin(), distances.end());
            return distances;
        }

        inline Histogram BOV::GetFrequencyHistogram(const FeatureInfo<float>& feature, 
            const vector<DescriptorInfo<float>>& words)
        {    
            int wordNum = words.size();
            int descriptorNum = feature.size();
            Histogram freqHistogram(wordNum, 0);

            for (int i = 0; i < descriptorNum; i++)
            {
                vector<double> distances = GetDistancesToVisualWords(feature[i], words);
                NormOneNormalize(distances.begin(), distances.end());

                for (int j = wordNum - 1; j >= 0; j--)
                    freqHistogram[j] += distances[j];
            }

            for (int i = 0; i < wordNum; i++)
                freqHistogram[i] /= descriptorNum;

            return freqHistogram;
        }

        inline vector<Histogram> BOV::GetFrequencyHistograms(const vector<FeatureInfo<float>>& features, 
            const vector<DescriptorInfo<float>>& words)
        {
            int imageNum = (int)features.size();
            vector<Histogram> freqHistograms(imageNum);

            #pragma omp parallel for
            for (int i = 0; i < imageNum; i++)
                freqHistograms[i] = GetFrequencyHistogram(features[i], words);

            return freqHistograms;
        }

        inline vector<Histogram> BOV::GetFrequencyHistogram(const vector<FeatureInfo<float>>& features,
            int clusterNum, int sampleNum)
        {
            return GetFrequencyHistograms(features, GetVisualWords(features, clusterNum, sampleNum));
        }
    }
}