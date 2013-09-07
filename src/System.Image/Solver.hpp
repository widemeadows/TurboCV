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

                evaIdxes = SplitDatasetEqually(labels, nFold);

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
        // APIs for LocalFeatureSolver
        //////////////////////////////////////////////////////////////////////////

        template<typename LocalFeature>
        class LocalFeatureSolver : public Solver
        {
        public:
            LocalFeatureSolver(
                cv::Mat (*preprocess)(const cv::Mat&), 
                const TString& datasetPath, 
                const TString& configFilePath = "config.xml"):
                Solver(preprocess, datasetPath, configFilePath) {}

            ArrayList<double> GetPrecisions() { return precisions; }
            ArrayList<Word_f> GetWords() { return words; }
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

            ArrayList<double> precisions;
            ArrayList<Word_f> words;
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

            double maxPrecision = -1;
            this->precisions.Clear();
            for (int i = 0; i < evaIdxes.Count(); i++)
            {
                printf("\nBegin Fold %d...\n", i + 1);
                const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
                ArrayList<LocalFeatureVec_f> trainingSet = Divide(features, pickUpIndexes).Item2();
                ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
                ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();

                printf("Compute Visual Words...\n");
                ArrayList<Word_f> words = BOV(SampleDescriptors(trainingSet, nSample), nWord).GetVisualWords();

                printf("Compute Frequency Histograms...\n");
                ArrayList<Histogram> histograms = FreqHist(features, words).GetFrequencyHistograms();
                ArrayList<Histogram> trainingHistograms = Divide(histograms, pickUpIndexes).Item2();
                ArrayList<Histogram> evaluationHistograms = Divide(histograms, pickUpIndexes).Item1();

                this->precisions.Add(KNN<Histogram>().
                    Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels).Item1());

                if (this->precisions[i] > maxPrecision)
                {
                    maxPrecision = precisions[i];
                    this->words = words;
                    this->histograms = histograms;
                }

                printf("Fold %d Accuracy: %f\n", i + 1, this->precisions[i]);
            }

            printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(precisions), 
                Math::StandardDeviation(precisions));
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
            this->precisions.Clear();
            for (int i = 0; i < evaIdxes.Count(); i++)
            {
                printf("\nBegin Fold %d...\n", i + 1);
                const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
                ArrayList<GlobalFeatureVec_f> trainingSet = Divide(features, pickUpIndexes).Item2();
                ArrayList<GlobalFeatureVec_f> evaluationSet = Divide(features, pickUpIndexes).Item1();
                ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
                ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();

                this->precisions.Add(KNN<GlobalFeatureVec_f>().
                    Evaluate(trainingSet, trainingLabels, evaluationSet, evaluationLabels).Item1());

                printf("Fold %d Accuracy: %f\n", i + 1, this->precisions[i]);
            }

            printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(precisions), 
                Math::StandardDeviation(precisions));
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
            this->precisions.Clear();
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

                this->precisions.Add(KNN<EdgeMatch>().
                    Evaluate(distToTraining, trainingLabels, evaluationLabels, HARD_VOTING).Item1());

                printf("Fold %d Accuracy: %f\n", i + 1, this->precisions[i]);
            }

            printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(precisions), 
                Math::StandardDeviation(precisions));
        }
    }
}
}