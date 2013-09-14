#pragma once

#include "../System/System.h"
#include "../System.XML/System.XML.h"
#include "Core.h"
#include <map>
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // APIs for Solvers
        //////////////////////////////////////////////////////////////////////////

        class Solver
        {
        public:
            Solver(
                cv::Mat (*preprocess)(const cv::Mat&), 
                const TString& datasetPath, 
                const TString& configFilePath = "config.xml")
            {
                this->datasetPath = datasetPath;
                this->configFilePath = configFilePath;
                this->Preprocess = preprocess;
            }

            void CrossValidation(int nFold = 3)
            {
                srand(1);
                
                auto dataset = LoadDataset(datasetPath);
                paths = dataset.Item1();
                labels = dataset.Item2();

                params = LoadConfiguration(configFilePath, AlgorithmName());

                evaIdxes = SplitDatasetRandomly(labels, nFold);

                InnerCrossValidation(paths, labels, params, evaIdxes);
            }

            ArrayList<TString> GetPaths() { return paths; }
            ArrayList<int> GetLabels() { return labels; }
            std::map<TString, TString> GetConfigParams() { return params; }
            ArrayList<ArrayList<size_t>> GetEvaIdxes() { return evaIdxes; }

        protected:
            cv::Mat (*Preprocess)(const cv::Mat&);

            virtual void InnerCrossValidation(
                const ArrayList<TString>& paths,
                const ArrayList<int>& labels,
                const std::map<TString, TString>& params,
                const ArrayList<ArrayList<size_t>>& evaIdxes) = 0;

            virtual TString AlgorithmName() const = 0;

            TString datasetPath;
            TString configFilePath;

            ArrayList<TString> paths;
            ArrayList<int> labels;

            std::map<TString, TString> params;

            ArrayList<ArrayList<size_t>> evaIdxes;
        };


        //////////////////////////////////////////////////////////////////////////
        // APIs for Enhanced LocalFeatureSolver
        //////////////////////////////////////////////////////////////////////////

        template<typename LocalFeature>
        class EnLocalFeatureSolver : public Solver
        {
        public:
            EnLocalFeatureSolver(
                cv::Mat (*preprocess)(const cv::Mat&), 
                const TString& datasetPath, 
                const TString& configFilePath = "config.xml",
                bool adaptiveSigma = false):
            Solver(preprocess, datasetPath, configFilePath), 
                adaptiveSigma(adaptiveSigma) {}

            ArrayList<double> GetAccuracies() { return accuracies; }
            ArrayList<double> GetPrecisions() { return precisions; }

            ArrayList<Word_f> GetWords() { return words; }
            ArrayList<double> GetSigmas() { return sigmas; }
            ArrayList<Histogram> GetHistograms() { return histograms; }

        protected:
            virtual void InnerCrossValidation(
                const ArrayList<TString>& paths,
                const ArrayList<int>& labels,
                const std::map<TString, TString>& params,
                const ArrayList<ArrayList<size_t>>& evaIdxes);

            virtual TString AlgorithmName() const
            {
                return LocalFeature().GetName();
            }

            bool adaptiveSigma;

            ArrayList<double> accuracies;
            ArrayList<double> precisions;

            ArrayList<Word_f> words;
            ArrayList<double> sigmas;
            ArrayList<Histogram> histograms;
        };

        template<typename LocalFeature>
        void EnLocalFeatureSolver<LocalFeature>::InnerCrossValidation(
            const ArrayList<TString>& paths,
            const ArrayList<int>& labels,
            const std::map<TString, TString>& params,
            const ArrayList<ArrayList<size_t>>& evaIdxes)
        {
            int nImage = paths.Count(), 
                nSample = GetDoubleValue(params, "inputNum", 1000000),
                nWord = GetDoubleValue(params, "wordNum", 500);

            printf("ImageNum: %d, SampleNum: %d, WordNum: %d\n", nImage, nSample, nWord);
            LocalFeature(params, true); // display all params of the algorithm

            printf("Cross Validation on " + LocalFeature().GetName() + "...\n");
            ArrayList<LocalFeatureVec_f> features(nImage);

            #pragma omp parallel for
            for (int i = 0; i < nImage; i++)
            {
                LocalFeature machine(params);
                cv::Mat image = cv::imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE); 
                Convert(machine(Preprocess != NULL ? Preprocess(image) : image), features[i]);
            }

            printf("Compute Visual Words...\n");
            ArrayList<Word_f> wordL1s = BOV(SampleDescriptors(features, nSample), nWord / 2).GetVisualWords();

            printf("Compute Frequency Histograms...\n");
            FreqHist* freqHist = new FreqHist(features, wordL1s);
            ArrayList<Histogram> histogramL1s = freqHist->GetFrequencyHistograms();

            printf("Compute Reconstructed Inputs...\n");
            features = freqHist->GetReconstructedInputs();

            delete freqHist;

            printf("Compute Visual Words...\n");
            ArrayList<Word_f> wordL2s = BOV(SampleDescriptors(features, nSample), nWord / 2).GetVisualWords();

            printf("Compute Frequency Histograms...\n");
            ArrayList<Histogram> histogramL2s = FreqHist(features, wordL2s).GetFrequencyHistograms();

            features.Clear();
            features.Shrink();

            ArrayList<Word_f> words;
            words.Add(wordL1s);
            words.Add(wordL2s);

            ArrayList<Histogram> histograms(nImage);
            for (int i = 0; i < nImage; i++)
            {
                histograms[i].Add(histogramL1s[i].begin(), histogramL1s[i].end());
                histograms[i].Add(histogramL2s[i].begin(), histogramL2s[i].end());

                for (int j = histograms[i].Count(); j >= 0; j--)
                    histograms[i][j] /= 2;
            }

            double maxAccuracy = -1;
            this->accuracies.Clear();
            for (int i = 0; i < evaIdxes.Count(); i++)
            {
                printf("\nBegin Fold %d...\n", i + 1);
                const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
                ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
                ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();
                ArrayList<Histogram> trainingHistograms = Divide(histograms, pickUpIndexes).Item2();
                ArrayList<Histogram> evaluationHistograms = Divide(histograms, pickUpIndexes).Item1();

                this->accuracies.Add(KNN<Histogram>().
                    Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels).Item1());

                if (this->accuracies[i] > maxAccuracy)
                {
                    maxAccuracy = accuracies[i];
                    this->words = words;
                    this->histograms = histograms;
                }

                printf("Fold %d Accuracy: %f\n", i + 1, this->accuracies[i]);
            }

            printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(this->accuracies), 
                Math::StandardDeviation(this->accuracies));

            this->precisions = MAP<Histogram>().
                FullCrossValidation(this->histograms, this->labels).Item1();

            printf("Mean Average Precision:");
            int nDisplay = (5 <= this->precisions.Count() ? 5 : this->precisions.Count());
            for (int i = 0; i < nDisplay; i++)
                printf(" %.4f", this->precisions[i]);
            printf("\n");
        }


        //////////////////////////////////////////////////////////////////////////
        // APIs for LocalFeatureSolver
        //////////////////////////////////////////////////////////////////////////

        template<typename LocalFeature>
        class LocalFeatureSolver : public Solver
        {
        public:
            LocalFeatureSolver(
                cv::Mat (*preprocess)(const cv::Mat&), 
                const TString& datasetPath, 
                const TString& configFilePath = "config.xml",
                bool adaptiveSigma = false):
                Solver(preprocess, datasetPath, configFilePath), 
                adaptiveSigma(adaptiveSigma) {}

            ArrayList<double> GetAccuracies() { return accuracies; }
            ArrayList<double> GetPrecisions() { return precisions; }

            ArrayList<Word_f> GetWords() { return words; }
            ArrayList<double> GetSigmas() { return sigmas; }
            ArrayList<Histogram> GetHistograms() { return histograms; }

        protected:
            virtual void InnerCrossValidation(
                const ArrayList<TString>& paths,
                const ArrayList<int>& labels,
                const std::map<TString, TString>& params,
                const ArrayList<ArrayList<size_t>>& evaIdxes);

            virtual TString AlgorithmName() const
            {
                return LocalFeature().GetName();
            }

            bool adaptiveSigma;

            ArrayList<double> accuracies;
            ArrayList<double> precisions;

            ArrayList<Word_f> words;
            ArrayList<double> sigmas;
            ArrayList<Histogram> histograms;
        };

        template<typename LocalFeature>
        void LocalFeatureSolver<LocalFeature>::InnerCrossValidation(
            const ArrayList<TString>& paths,
            const ArrayList<int>& labels,
            const std::map<TString, TString>& params,
            const ArrayList<ArrayList<size_t>>& evaIdxes)
        {
            int nImage = paths.Count(), 
                nSample = GetDoubleValue(params, "inputNum", 1000000),
                nWord = GetDoubleValue(params, "wordNum", 500);

            printf("ImageNum: %d, SampleNum: %d, WordNum: %d\n", nImage, nSample, nWord);
            LocalFeature(params, true); // display all params of the algorithm

            printf("Cross Validation on " + LocalFeature().GetName() + "...\n");
            ArrayList<LocalFeatureVec_f> features(nImage);

            #pragma omp parallel for
            for (int i = 0; i < nImage; i++)
            {
                LocalFeature machine(params);
                cv::Mat image = cv::imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE); 
                Convert(machine(Preprocess != NULL ? Preprocess(image) : image), features[i]);
            }

            double maxAccuracy = -1;
            this->accuracies.Clear();
            for (int i = 0; i < evaIdxes.Count(); i++)
            {
                printf("\nBegin Fold %d...\n", i + 1);
                const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
                ArrayList<LocalFeatureVec_f> trainingSet = Divide(features, pickUpIndexes).Item2();
                ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
                ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();

                printf("Compute Visual Words...\n");
                BOV bov(SampleDescriptors(trainingSet, nSample), nWord);
                ArrayList<Word_f> words = bov.GetVisualWords();
                ArrayList<double> sigmas;

                printf("Compute Frequency Histograms...\n");
                ArrayList<Histogram> histograms;

                if (adaptiveSigma)
                {
                    sigmas = bov.GetSigmas();
                    histograms = FreqHist(features, words, sigmas).GetFrequencyHistograms();
                }
                else
                {
                    FreqHist freqHist(features, words);
                    histograms = freqHist.GetFrequencyHistograms();
                    sigmas = freqHist.GetSigmas();
                }
                
                ArrayList<Histogram> trainingHistograms = Divide(histograms, pickUpIndexes).Item2();
                ArrayList<Histogram> evaluationHistograms = Divide(histograms, pickUpIndexes).Item1();

                this->accuracies.Add(KNN<Histogram>().
                    Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels).Item1());

                if (this->accuracies[i] > maxAccuracy)
                {
                    maxAccuracy = accuracies[i];
                    this->words = words;
                    this->sigmas = sigmas;
                    this->histograms = histograms;
                }

                printf("Fold %d Accuracy: %f\n", i + 1, this->accuracies[i]);
            }

            printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(this->accuracies), 
                Math::StandardDeviation(this->accuracies));

            this->precisions = MAP<Histogram>().
                FullCrossValidation(this->histograms, this->labels).Item1();

            printf("Mean Average Precision:");
            int nDisplay = (5 <= this->precisions.Count() ? 5 : this->precisions.Count());
            for (int i = 0; i < nDisplay; i++)
                printf(" %.4f", this->precisions[i]);
            printf("\n");
        }


        //////////////////////////////////////////////////////////////////////////
        // APIs for GlobalFeatureSolver
        //////////////////////////////////////////////////////////////////////////

        template<typename GlobalFeature>
        class GlobalFeatureSolver : public Solver
        {
        public:
            GlobalFeatureSolver(
                cv::Mat (*preprocess)(const cv::Mat&), 
                const TString& datasetPath, 
                const TString& configFilePath = "config.xml"):
                Solver(preprocess, datasetPath, configFilePath) {}

            ArrayList<double> GetAccuracies() { return accuracies; }
            ArrayList<double> GetPrecisions() { return precisions; }

            ArrayList<GlobalFeatureVec_f> GetFeatures() { return features; }

        protected:
            virtual void InnerCrossValidation(
                const ArrayList<TString>& paths,
                const ArrayList<int>& labels,
                const std::map<TString, TString>& params,
                const ArrayList<ArrayList<size_t>>& evaIdxes);

            virtual TString AlgorithmName() const
            {
                return GlobalFeature().GetName();
            }

        private:
            ArrayList<double> accuracies;
            ArrayList<double> precisions;

            ArrayList<GlobalFeatureVec_f> features;
        };

        template<typename GlobalFeature>
        void GlobalFeatureSolver<GlobalFeature>::InnerCrossValidation(
            const ArrayList<TString>& paths,
            const ArrayList<int>& labels,
            const std::map<TString, TString>& params,
            const ArrayList<ArrayList<size_t>>& evaIdxes)
        {
            int nImage = paths.Count();

            printf("ImageNum: %d\n", nImage);
            GlobalFeature(params, true); // display all params of the algorithm

            printf("Cross Validation on " + GlobalFeature().GetName() + "...\n");
            ArrayList<GlobalFeatureVec_f> features(nImage);

            #pragma omp parallel for
            for (int i = 0; i < nImage; i++)
            {
                GlobalFeature machine(params);
                cv::Mat image = cv::imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE); 
                Convert(machine(Preprocess != NULL ? Preprocess(image) : image), features[i]);
            }

            this->features = features;

            double maxPrecision = -1;
            this->accuracies.Clear();
            for (int i = 0; i < evaIdxes.Count(); i++)
            {
                printf("\nBegin Fold %d...\n", i + 1);
                const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
                ArrayList<GlobalFeatureVec_f> trainingSet = Divide(features, pickUpIndexes).Item2();
                ArrayList<GlobalFeatureVec_f> evaluationSet = Divide(features, pickUpIndexes).Item1();
                ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
                ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();

                this->accuracies.Add(KNN<GlobalFeatureVec_f>().
                    Evaluate(trainingSet, trainingLabels, evaluationSet, evaluationLabels).Item1());

                printf("Fold %d Accuracy: %f\n", i + 1, this->accuracies[i]);
            }

            printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(this->accuracies), 
                Math::StandardDeviation(this->accuracies));

            this->precisions = MAP<GlobalFeatureVec_f>().
                FullCrossValidation(this->features, this->labels).Item1();

            printf("Mean Average Precision:");
            int nDisplay = (5 <= this->precisions.Count() ? 5 : this->precisions.Count());
            for (int i = 0; i < nDisplay; i++)
                printf(" %.4f", this->precisions[i]);
            printf("\n");
        }


        //////////////////////////////////////////////////////////////////////////
        // APIs for EdgeMatchSolver
        //////////////////////////////////////////////////////////////////////////

        template<typename EdgeMatch>
        class EdgeMatchSolver : public Solver
        {
        public:
            EdgeMatchSolver(
                cv::Mat (*preprocess)(const cv::Mat&), 
                const TString& datasetPath, 
                const TString& configFilePath = "config.xml"):
                Solver(preprocess, datasetPath, configFilePath) {}

            ArrayList<double> GetAccuracies() { return accuracies; }
            ArrayList<double> GetPrecisions() { return precisions; }

            cv::Mat GetDistanceMatrix() { return distanceMatrix; }

        protected:
            virtual void InnerCrossValidation(
                const ArrayList<TString>& paths,
                const ArrayList<int>& labels,
                const std::map<TString, TString>& params,
                const ArrayList<ArrayList<size_t>>& evaIdxes);

            virtual TString AlgorithmName() const
            {
                return EdgeMatch().GetName();
            }

        private:
            ArrayList<double> accuracies;
            ArrayList<double> precisions;

            cv::Mat distanceMatrix;
        };

        template<typename EdgeMatch>
        void EdgeMatchSolver<EdgeMatch>::InnerCrossValidation(
            const ArrayList<TString>& paths,
            const ArrayList<int>& labels,
            const std::map<TString, TString>& params,
            const ArrayList<ArrayList<size_t>>& evaIdxes)
        {
            int nImage = paths.Count();

            printf("ImageNum: %d\n", nImage);
            EdgeMatch(params, true); // display all params of the algorithm

            printf("Cross Validation on " + EdgeMatch().GetName() + "...\n");
            
            ArrayList<cv::Mat> images(nImage);
            ArrayList<ArrayList<PointList>> channels(nImage);
            #pragma omp parallel for
            for (int i = 0; i < nImage; i++)
            {
                EdgeMatch machine(params);
                cv::Mat image = cv::imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE); 
                images[i] = (Preprocess != NULL ? Preprocess(image) : image);
                channels[i] = machine.GetChannels(images[i]);
            }

            cv::Mat oneWayDistMat(nImage, nImage, CV_64F);
            #pragma omp parallel for
            for (int i = 0; i < nImage; i++)
            {
                EdgeMatch machine(params);
                ArrayList<cv::Mat> transforms = machine.GetTransforms(images[i].size(), channels[i]);

                for (int j = 0; j < nImage; j++)
                {
                    if (j == i)
                        oneWayDistMat.at<double>(i, j) = 0;
                    else
                        oneWayDistMat.at<double>(i, j) = machine.GetOneWayDistance(channels[j], transforms);
                }
            }

            cv::Mat twoWayDistMat(nImage, nImage, CV_64F);
            #pragma omp parallel for
            for (int i = 0; i < nImage; i++)
            {
                EdgeMatch machine(params);

                for (int j = 0; j < nImage; j++)
                {
                    if (j == i)
                        twoWayDistMat.at<double>(i, j) = 0;
                    else
                        twoWayDistMat.at<double>(i, j) = machine.GetTwoWayDistance(
                            oneWayDistMat.at<double>(i, j), oneWayDistMat.at<double>(j, i));
                }
            }
            this->distanceMatrix = twoWayDistMat;

            double maxPrecision = -1;
            this->accuracies.Clear();
            for (int i = 0; i < evaIdxes.Count(); i++)
            {
                printf("\nBegin Fold %d...\n", i + 1);
                const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
                ArrayList<size_t> restIndexes = Rest(nImage, pickUpIndexes);
                ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
                ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();
                int trainingNum = trainingLabels.Count(), evaluationNum = evaluationLabels.Count();

                cv::Mat distToTraining(evaluationNum, trainingNum, CV_64F);
                for (int i = 0; i < evaluationNum; i++)
                    for (int j = 0; j < trainingNum; j++)
                        distToTraining.at<double>(i, j) = this->distanceMatrix.at<double>(
                            pickUpIndexes[i], restIndexes[j]);

                this->accuracies.Add(KNN<EdgeMatch>().
                    Evaluate(distToTraining, trainingLabels, evaluationLabels, HARD_VOTING).Item1());

                printf("Fold %d Accuracy: %f\n", i + 1, this->accuracies[i]);
            }

            printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(this->accuracies), 
                Math::StandardDeviation(this->accuracies));

            this->precisions = MAP<Histogram>().
                FullCrossValidation(this->distanceMatrix, this->labels).Item1();

            printf("Mean Average Precision:");
            int nDisplay = (5 <= this->precisions.Count() ? 5 : this->precisions.Count());
            for (int i = 0; i < nDisplay; i++)
                printf(" %.4f", this->precisions[i]);
            printf("\n");
        }
    }
}
}