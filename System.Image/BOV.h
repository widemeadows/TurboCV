#pragma once

#include "../System/System.h"
#include "Feature.h"
#include "Typedef.h"
#include <cv.h>
using namespace cv;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        class BOV
        {
        public:
            static ArrayList<Word_f> GetVisualWords(
                const ArrayList<LocalFeature_f>& features, 
                size_t clusterNum, 
                size_t sampleNum = INF);

            static ArrayList<Histogram> GetFrequencyHistograms(
                const ArrayList<LocalFeature_f>& features, 
                const ArrayList<Word_f>& words);

            static ArrayList<Histogram> GetFrequencyHistogram(
                const ArrayList<LocalFeature_f>& features, 
                size_t clusterNum, 
                size_t sampleNum = INF);

            static Histogram BOV::GetFrequencyHistogram(
                const LocalFeature_f& feature, 
                const ArrayList<Word_f>& words);
        
        private:
            static ArrayList<double> GetDistancesToVisualWords(
                const Descriptor_f& descriptor, 
                const ArrayList<Word_f>& words);
        };

        inline ArrayList<Word_f> BOV::GetVisualWords(
            const ArrayList<LocalFeature_f>& features, 
            size_t clusterNum, 
            size_t sampleNum)
        {
            ArrayList<Descriptor_f> allDescriptors;
            for (size_t i = 0; i < features.size(); i++)
                for (size_t j = 0; j < features[i].size(); j++)
                    allDescriptors.push_back(features[i][j]);

            assert(allDescriptors.size() > 0);
            size_t descriptorNum = allDescriptors.size(), 
                descriptorSize = allDescriptors[0].size();
            printf("Descriptor Num: %d, Descriptor Size: %d.\n", 
                (int)descriptorNum, (int)descriptorSize);

            sampleNum = min(descriptorNum, sampleNum);
            Mat samples(sampleNum, descriptorSize, CV_32F);
            ArrayList<size_t> randomIndex = RandomPermutate(descriptorNum, sampleNum);
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

            ArrayList<Word_f> words(clusterNum);
            for (size_t i = 0; i < clusterNum; i++)
                for (size_t j = 0; j < descriptorSize; j++)
                    words[i].push_back(centers.at<float>(i, j));

            return words;
        }

        inline ArrayList<double> BOV::GetDistancesToVisualWords(
            const Descriptor_f& descriptor, 
            const ArrayList<Word_f>& words)
        {
            assert(words.size() > 0 && descriptor.size() == words[0].size());

            size_t wordNum = words.size();
            ArrayList<double> distances;

            for (size_t i = 0; i < wordNum; i++)
                distances.push_back(Math::GaussianDistance(descriptor, words[i], 0.1));

            NormOneNormalize(distances.begin(), distances.end());
            return distances;
        }

        inline Histogram BOV::GetFrequencyHistogram(
            const LocalFeature_f& feature, 
            const ArrayList<Word_f>& words)
        {    
            size_t wordNum = words.size();
            size_t descriptorNum = feature.size();
            Histogram freqHistogram(wordNum);

            for (size_t i = 0; i < descriptorNum; i++)
            {
                ArrayList<double> distances = GetDistancesToVisualWords(feature[i], words);
                NormOneNormalize(distances.begin(), distances.end());

                for (size_t j = 0; j < wordNum; j++)
                    freqHistogram[j] += distances[j];
            }

            for (size_t i = 0; i < wordNum; i++)
                freqHistogram[i] /= descriptorNum;

            return freqHistogram;
        }

        inline ArrayList<Histogram> BOV::GetFrequencyHistograms(
            const ArrayList<LocalFeature_f>& features, 
            const ArrayList<Word_f>& words)
        {
            size_t imageNum = features.size();
            ArrayList<Histogram> freqHistograms(imageNum);

            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < imageNum; i++)
                freqHistograms[i] = GetFrequencyHistogram(features[i], words);

            return freqHistograms;
        }

        inline ArrayList<Histogram> BOV::GetFrequencyHistogram(
            const ArrayList<LocalFeature_f>& features,
            size_t clusterNum, size_t sampleNum)
        {
            return GetFrequencyHistograms(features, 
                GetVisualWords(features, clusterNum, sampleNum));
        }

        //class IDF
        //{
        //public:
        //    static vector<double> GetWeights(const vector<Histogram>& histograms)
        //    {
        //        if (histograms.size() == 0)
        //            return vector<double>();

        //        size_t histNum = histograms.size(), histSize = histograms[0].size();
        //        vector<double> mean(histSize), deviation(histSize);

        //        for (size_t i = 0; i < histNum; i++)
        //        {
        //            for (size_t j = 0; j < histSize; j++)
        //            {
        //                mean[j] += histograms[i][j];
        //                deviation[j] += histograms[i][j] * histograms[i][j];
        //            }
        //        }

        //        for (size_t i = 0; i < histSize; i++)
        //        {
        //            mean[i] /= histNum;
        //            deviation[i] = sqrt((deviation[i] - histNum * mean[i] * mean[i]) / histNum);
        //        }

        //        vector<pair<double, int>> sorted(histSize);
        //        for (size_t i = 0; i < histSize; i++)
        //            sorted[i] = make_pair(deviation[i], i);

        //        sort(sorted.begin(), sorted.end());

        //        for (size_t i = 0; i < 200; i++)
        //            deviation[sorted[i].second] = 0;

        //        return deviation;
        //    }
        //};
    }
}
}