#pragma once

#include "../System/System.h"
#include "Feature.h"
#include "Typedef.h"
#include <cv.h>
using namespace std;

namespace System
{
    namespace Image
    {
        class BOV
        {
        public:
            static vector<Word_f> GetVisualWords(
                const vector<LocalFeature_f>& features, size_t clusterNum, size_t sampleNum = INF);

            static vector<Histogram> GetFrequencyHistograms(
                const vector<LocalFeature_f>& features, const vector<Word_f>& words);

            static vector<Histogram> GetFrequencyHistogram(
                const vector<LocalFeature_f>& features, size_t clusterNum, size_t sampleNum = INF);
        
        private:
            static vector<double> GetDistancesToVisualWords(const Descriptor_f& descriptor, 
                const vector<Word_f>& words);

            static Histogram BOV::GetFrequencyHistogram(const LocalFeature_f& feature, 
                const vector<Word_f>& words);
        };

        inline vector<Word_f> BOV::GetVisualWords(
            const vector<LocalFeature_f>& features, size_t clusterNum, size_t sampleNum)
        {
            vector<Descriptor_f> allDescriptors;
            for (size_t i = 0; i < features.size(); i++)
                for (size_t j = 0; j < features[i].size(); j++)
                    allDescriptors.push_back(features[i][j]);

            assert(allDescriptors.size() > 0);
            size_t descriptorNum = allDescriptors.size(), descriptorSize = allDescriptors[0].size();
            printf("Descriptor Num: %d, Descriptor Size: %d.\n", (int)descriptorNum, (int)descriptorSize);

            sampleNum = min(descriptorNum, sampleNum);
            Mat samples(sampleNum, descriptorSize, CV_32F);
            vector<size_t> randomIndex = RandomPermutate(descriptorNum, sampleNum);
            sort(randomIndex.begin(), randomIndex.end());

            int counter = 0;
            for (size_t i = 0; i < randomIndex.size(); i++)
            {
                size_t index = randomIndex[i];

                for (size_t j = 0; j < descriptorSize; j++)
                    samples.at<float>(counter, j) = allDescriptors[index][j];

                counter++;
            }

            Mat labels(sampleNum, 1, CV_32S);
            for (size_t i = 0; i < sampleNum; i++)
                labels.at<int>(i, 0) = 0;

            Mat centers(clusterNum, descriptorSize, CV_32F);
            printf("K-Means Begin...\n");
            kmeans(samples, clusterNum, labels, TermCriteria(CV_TERMCRIT_ITER, 500, 1e-6), 
                1, KMEANS_PP_CENTERS, centers);

            vector<Word_f> words(clusterNum);
            for (size_t i = 0; i < clusterNum; i++)
                for (size_t j = 0; j < descriptorSize; j++)
                    words[i].push_back(centers.at<float>(i, j));

            return words;
        }

        inline vector<double> BOV::GetDistancesToVisualWords(
            const Descriptor_f& descriptor, const vector<Word_f>& words)
        {
            assert(words.size() > 0 && descriptor.size() == words[0].size());

            size_t wordNum = words.size();
            vector<double> distances;

            for (size_t i = 0; i < wordNum; i++)
                distances.push_back(Math::GaussianDistance(descriptor, words[i], 0.1));

            NormOneNormalize(distances.begin(), distances.end());
            return distances;
        }

        inline Histogram BOV::GetFrequencyHistogram(const LocalFeature_f& feature, 
            const vector<Word_f>& words)
        {    
            size_t wordNum = words.size();
            size_t descriptorNum = feature.size();
            Histogram freqHistogram(wordNum);

            for (size_t i = 0; i < descriptorNum; i++)
            {
                vector<double> distances = GetDistancesToVisualWords(feature[i], words);
                NormOneNormalize(distances.begin(), distances.end());

                for (size_t j = 0; j < wordNum; j++)
                    freqHistogram[j] += distances[j];
            }

            for (size_t i = 0; i < wordNum; i++)
                freqHistogram[i] /= descriptorNum;

            return freqHistogram;
        }

        inline vector<Histogram> BOV::GetFrequencyHistograms(const vector<LocalFeature_f>& features, 
            const vector<Word_f>& words)
        {
            size_t imageNum = features.size();
            vector<Histogram> freqHistograms(imageNum);

            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < imageNum; i++)
                freqHistograms[i] = GetFrequencyHistogram(features[i], words);

            return freqHistograms;
        }

        inline vector<Histogram> BOV::GetFrequencyHistogram(const vector<LocalFeature_f>& features,
            size_t clusterNum, size_t sampleNum)
        {
            return GetFrequencyHistograms(features, GetVisualWords(features, clusterNum, sampleNum));
        }

        class IDF
        {
        public:
            static vector<double> GetWeights(const vector<Histogram>& histograms)
            {
                if (histograms.size() == 0)
                    return vector<double>();

                size_t histNum = histograms.size(), histSize = histograms[0].size();
                vector<double> mean(histSize), deviation(histSize);

                for (size_t i = 0; i < histNum; i++)
                {
                    for (size_t j = 0; j < histSize; j++)
                    {
                        mean[j] += histograms[i][j];
                        deviation[j] += histograms[i][j] * histograms[i][j];
                    }
                }

                for (size_t i = 0; i < histSize; i++)
                {
                    mean[i] /= histNum;
                    deviation[i] = sqrt((deviation[i] - histNum * mean[i] * mean[i]) / histNum);
                }

                return deviation;
            }
        };
    }
}