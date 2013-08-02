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
            size_t clusterNum,
            cv::TermCriteria termCriteria)
        {
            assert(descriptors.Count() > 0 && descriptors[0].Count() > 0);

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
            kmeans(samples, clusterNum, labels, termCriteria, 1, KMEANS_PP_CENTERS, centers);

            ArrayList<Word_f> words(clusterNum);
            for (size_t i = 0; i < clusterNum; i++)
                for (size_t j = 0; j < descriptorSize; j++)
                    words[i].Add(centers.at<float>(i, j));

            return words;
        }

        ArrayList<double> BOV::GetSigmas(
            const ArrayList<Descriptor_f>& descriptors,
            const ArrayList<Word_f>& words,
            double perplexity)
        {
            assert(descriptors.Count() > 0 && words.Count() > 0);

            size_t descriptorNum = descriptors.Count();
            size_t wordNum = words.Count();
            const double logU = std::log(perplexity);
            const double INF = 1e12;
            const double EPS = 1e-5;
            ArrayList<double> sigmas(wordNum);
            Mat D(wordNum, descriptorNum, CV_64F);

            #pragma omp parallel for
            for (int i = 0; i < wordNum; i++)
                for (int j = 0; j < descriptorNum; j++)
                    D.at<double>(i, j) = Math::NormTwoDistance(words[i], descriptors[j]);

            #pragma omp parallel for
            for (int i = 0; i < wordNum; i++)
            {
                double lowerBound = -INF;
                double upperBound = INF;

                Mat Di = D.row(i).clone(); // square of L2 distance
                Mat GDi(Di.size(), CV_64F); // gaussian distance
                double Si = 1.0;
                double sumGDi = 0, H = 0, Hdiff = 0;

                for (int tries = 0; tries < 50; tries++) 
                {
                    exp(Di / -Si, GDi);
                    sumGDi = sum(GDi)[0];

                    H = Di.dot(GDi) / (sumGDi * Si) + std::log(sumGDi);
                    Hdiff = logU - H;

                    if (Hdiff > 0)
                    {
                        lowerBound = Si;

                        if (upperBound == INF)
                            Si *= 2;
                        else
                            Si = (Si + upperBound) / 2;
                    }
                    else
                    {
                        upperBound = Si;

                        if (lowerBound == -INF)
                            Si /= 2;
                        else
                            Si = (Si + lowerBound) / 2;
                    }

                    if (std::abs(Hdiff) < EPS)
                        break;
                }

                sigmas[i] = std::sqrt(Si / 2);
            }

            return sigmas;
        }


        //////////////////////////////////////////////////////////////////////////
        // FrequencyHistogram
        //////////////////////////////////////////////////////////////////////////

        ArrayList<Histogram> FreqHist::GetFrequencyHistograms(
            const ArrayList<LocalFeatureVec_f>& features, 
            const ArrayList<Word_f>& words)
        {
            size_t imageNum = features.Count();
            ArrayList<Histogram> freqHistograms(imageNum);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < imageNum; i++)
                freqHistograms[i] = GetFrequencyHistogram(features[i], words);

            return freqHistograms;
        }

        Histogram FreqHist::GetFrequencyHistogram(
            const LocalFeatureVec_f& feature, 
            const ArrayList<Word_f>& words)
        {    
            size_t wordNum = words.Count();
            size_t descriptorNum = feature.Count();
            Histogram freqHistogram(wordNum);

            for (size_t i = 0; i < descriptorNum; i++)
            {
                ArrayList<double> distances = GetDistancesToVisualWords(feature[i], words);
                for (size_t j = 0; j < wordNum; j++)
                    freqHistogram[j] += distances[j];
            }

            for (size_t i = 0; i < wordNum; i++)
                freqHistogram[i] /= descriptorNum;

            return freqHistogram;
        }

        ArrayList<LocalFeatureVec> FreqHist::GetPoolingFeatures(
            const ArrayList<LocalFeatureVec_f>& features, 
            const ArrayList<Word_f>& words,
            int nPool)
        {
            size_t imageNum = features.Count();
            ArrayList<LocalFeatureVec> poolFeatures(imageNum);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < imageNum; i++)
                poolFeatures[i] = GetPoolingFeature(features[i], words, nPool);

            return poolFeatures;
        }

        LocalFeatureVec FreqHist::GetPoolingFeature(
            const LocalFeatureVec_f& feature, 
            const ArrayList<Word_f>& words, 
            int nPool)
        {
            size_t wordNum = words.Count();
            size_t descriptorNum = feature.Count();
            size_t descriptorNumPerSize = (size_t)sqrt(descriptorNum);
            size_t poolStep = descriptorNumPerSize / nPool;
            LocalFeatureVec poolFeature(nPool * nPool);
            assert(descriptorNumPerSize * descriptorNumPerSize == descriptorNum);

            for (size_t i = 0; i < descriptorNum; i++)
            {
                Histogram histogram(wordNum);

                ArrayList<double> distances = GetDistancesToVisualWords(feature[i], words);
                for (size_t j = 0; j < wordNum; j++)
                    histogram[j] += distances[j];

                int row = i / descriptorNumPerSize, col = i % descriptorNumPerSize;
                int poolIdx = (row / poolStep) * nPool + (col / poolStep);
                assert(0 <= poolIdx && poolIdx < poolFeature.Count());
                poolFeature[poolIdx] = histogram;
            }

            for (size_t i = 0; i < poolFeature.Count(); i++)
                for (size_t j = 0; j < wordNum; j++)
                    poolFeature[i][j] /= poolStep * poolStep;

            return poolFeature;
        }

        ArrayList<double> FreqHist::GetDistancesToVisualWords(
            const Descriptor_f& descriptor, 
            const ArrayList<Word_f>& words)
        {
            assert(words.Count() > 0 && descriptor.Count() == words[0].Count());

            size_t wordNum = words.Count();
            ArrayList<double> distances;

            for (size_t i = 0; i < wordNum; i++)
                distances.Add(getDistance(descriptor, words[i]));

            NormOneNormalize(distances.begin(), distances.end());
            return distances;
        }
    }
}
}