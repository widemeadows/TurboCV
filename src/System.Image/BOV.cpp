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

            #pragma omp parallel for
            for (int i = 0; i < wordNum; i++)
            {
                double lowerBound = -INF;
                double upperBound = INF;

                Mat Di(1, descriptorNum, CV_64F); // square of L2 distance
                Mat GDi(1, descriptorNum, CV_64F); // gaussian distance
                double Si = 1.0;
                double sumGDi = 0, H = 0, Hdiff = 0;

                for (int j = 0; j < descriptorNum; j++)
                    Di.at<double>(0, j) = Math::NormTwoDistance(words[i], descriptors[j]);

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

        ArrayList<Histogram> FreqHist::ComputeFrequencyHistograms()
        {
            size_t imageNum = features.Count();
            ArrayList<Histogram> freqHistograms(imageNum);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < imageNum; i++)
                freqHistograms[i] = ComputeFrequencyHistogram(features[i]);

            return freqHistograms;
        }

        Histogram FreqHist::ComputeFrequencyHistogram(const LocalFeatureVec_f& feature)
        {    
            size_t wordNum = words.Count();
            size_t descriptorNum = feature.Count();
            Histogram freqHistogram(wordNum);

            for (size_t i = 0; i < descriptorNum; i++)
            {
                ArrayList<double> distances = GetDistancesToVisualWords(feature[i]);
                for (size_t j = 0; j < wordNum; j++)
                    freqHistogram[j] += distances[j];
            }

            for (size_t i = 0; i < wordNum; i++)
                freqHistogram[i] /= descriptorNum;

            return freqHistogram;
        }

        ArrayList<LocalFeatureVec> FreqHist::ComputePoolingHistograms(int nPool)
        {
            size_t imageNum = features.Count();
            ArrayList<LocalFeatureVec> poolFeatures(imageNum);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < imageNum; i++)
                poolFeatures[i] = ComputePoolingHistogram(features[i], nPool);

            return poolFeatures;
        }

        LocalFeatureVec FreqHist::ComputePoolingHistogram(const LocalFeatureVec_f& feature, int nPool)
        {
            size_t wordNum = words.Count();
            size_t descriptorNum = feature.Count();
            size_t descriptorNumPerSize = (size_t)sqrt(descriptorNum);
            size_t poolStep = descriptorNumPerSize / nPool;
            LocalFeatureVec poolFeature(nPool * nPool);
            assert(descriptorNumPerSize * descriptorNumPerSize == descriptorNum);

            for (size_t i = 0; i < poolFeature.Count(); i++)
            {
                poolFeature[i] = Histogram(wordNum);
            }

            for (size_t i = 0; i < descriptorNum; i++)
            {
                int row = i / descriptorNumPerSize, col = i % descriptorNumPerSize;
                int poolIdx = (row / poolStep) * nPool + (col / poolStep);
                assert(0 <= poolIdx && poolIdx < poolFeature.Count());

                ArrayList<double> distances = GetDistancesToVisualWords(feature[i]);
                for (size_t j = 0; j < wordNum; j++)
                    poolFeature[poolIdx][j] += distances[j];
            }

            for (size_t i = 0; i < poolFeature.Count(); i++)
                for (size_t j = 0; j < wordNum; j++)
                    poolFeature[i][j] /= poolStep * poolStep;

            return poolFeature;
        }

        ArrayList<LocalFeatureVec_f> FreqHist::ComputeReconstructedInputs()
        {
            size_t imageNum = features.Count();
            ArrayList<LocalFeatureVec_f> reconstructed(imageNum);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < imageNum; i++)
                reconstructed[i] = ComputeReconstructedInput(features[i]);

            return reconstructed;
        }

        LocalFeatureVec_f FreqHist::ComputeReconstructedInput(const LocalFeatureVec_f& feature)
        {
            size_t wordNum = words.Count();
            size_t descriptorNum = feature.Count();
            size_t descriptorSize = feature[0].Count();
            LocalFeatureVec reconstructed;

            for (size_t i = 0; i < descriptorNum; i++)
            {
                ArrayList<double> distances = GetDistancesToVisualWords(feature[i]);

                Descriptor reDescriptor(descriptorSize);
                for (size_t j = 0; j < descriptorSize; i++)
                    for (size_t k = 0; k < wordNum; k++)
                        reDescriptor[j] += distances[k] * words[k][j];
                
                reconstructed.Add(reDescriptor);
            }

            LocalFeatureVec_f result;
            Convert(reconstructed, result);
            return result;
        }

        ArrayList<double> FreqHist::GetDistancesToVisualWords(const Descriptor_f& descriptor)
        {
            assert(words.Count() > 0 && words.Count() == sigmas.Count() &&
                descriptor.Count() == words[0].Count());

            size_t wordNum = words.Count();
            ArrayList<double> distances(wordNum);

            for (size_t i = 0; i < wordNum; i++)
                distances[i] = Math::GaussianDistance(descriptor, words[i], sigmas[i]);

            NormOneNormalize(distances.begin(), distances.end());
            return distances;
        }
    }
}
}