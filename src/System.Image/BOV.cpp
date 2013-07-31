#include "../System/System.h"
#include "Core.h"
#include <cv.h>
using namespace cv;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // BOV
        //////////////////////////////////////////////////////////////////////////

        ArrayList<Word_f> BOV::GetVisualWords(
            const ArrayList<Descriptor_f>& descriptors, 
            size_t clusterNum)
        {
            assert(descriptors.Count() > 0);
            size_t descriptorNum = descriptors.Count(), 
                descriptorSize = descriptors[0].Count();
            printf("Descriptor Num: %d, Descriptor Size: %d.\n", 
                (int)descriptorNum, (int)descriptorSize);

            Mat samples(descriptorNum, descriptorSize, CV_32F);
            for (size_t i = 0; i < descriptorNum; i++)
                for (size_t j = 0; j < descriptorSize; j++)
                    samples.at<float>(i, j) = descriptors[i][j];

            Mat labels(descriptorNum, 1, CV_32S);
            for (size_t i = 0; i < descriptorNum; i++)
                labels.at<int>(i, 0) = 0;

            Mat centers(clusterNum, descriptorSize, CV_32F);
            printf("K-Means Begin...\n");
            kmeans(samples, clusterNum, labels, TermCriteria(CV_TERMCRIT_ITER, 500, 1e-6), 
                1, KMEANS_PP_CENTERS, centers);

            ArrayList<Word_f> words(clusterNum);
            for (size_t i = 0; i < clusterNum; i++)
                for (size_t j = 0; j < descriptorSize; j++)
                    words[i].Add(centers.at<float>(i, j));

            return words;
        }


        //////////////////////////////////////////////////////////////////////////
        // FrequencyHistogram
        //////////////////////////////////////////////////////////////////////////

        Group<ArrayList<Histogram>, ArrayList<LocalFeatureVec>> FreqHist::GetFrequencyHistograms(
            const ArrayList<LocalFeatureVec_f>& features, 
            const ArrayList<Word_f>& words)
        {
            size_t imageNum = features.Count();
            ArrayList<Histogram> freqHistograms(imageNum);
            ArrayList<LocalFeatureVec> distances(imageNum);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < imageNum; i++)
            {
                auto result = GetFrequencyHistogram(features[i], words);
                freqHistograms[i] = result.Item1();
                distances[i] = result.Item2();
            }

            return CreateGroup(freqHistograms, distances);
        }

        Group<Histogram, LocalFeatureVec> FreqHist::GetFrequencyHistogram(
            const LocalFeatureVec_f& feature, 
            const ArrayList<Word_f>& words)
        {    
            size_t wordNum = words.Count();
            size_t descriptorNum = feature.Count();
            Histogram freqHistogram(wordNum);
            LocalFeatureVec allDistances(descriptorNum);

            for (size_t i = 0; i < descriptorNum; i++)
            {
                ArrayList<double> distances = GetDistancesToVisualWords(feature[i], words);

                for (size_t j = 0; j < wordNum; j++)
                    freqHistogram[j] += distances[j];
                
                allDistances[i] = distances;
            }

            for (size_t i = 0; i < wordNum; i++)
                freqHistogram[i] /= descriptorNum;

            return CreateGroup(freqHistogram, allDistances);
        }

        ArrayList<double> FreqHist::GetDistancesToVisualWords(
            const Descriptor_f& descriptor, 
            const ArrayList<Word_f>& words)
        {
            assert(words.Count() > 0 && descriptor.Count() == words[0].Count());

            size_t wordNum = words.Count();
            ArrayList<double> distances;

            for (size_t i = 0; i < wordNum; i++)
                distances.Add(Math::GaussianDistance(descriptor, words[i], 0.1));

            NormOneNormalize(distances.begin(), distances.end());
            return distances;
        }
    }
}
}