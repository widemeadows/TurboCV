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

            void LoadData()
            {
                doc = LoadConfiguration(configFilePath);

                auto dataset = LoadDataset(datasetPath);
                paths = dataset.Item1();
                labels = dataset.Item2();
            }

            static System::XML::TiXmlDocument LoadConfiguration(const TString& configFilePath);
            static Group<ArrayList<TString>, ArrayList<int>> LoadDataset(const TString& datasetPath);

            std::map<TString, TString> GetConfiguration(const TString& featureName);
            ArrayList<TString> GetPaths() { return paths; }
            ArrayList<int> GetLabels() { return labels; }

            virtual void CrossValidation(int nFold = 3) = 0;

        protected:
            cv::Mat (*Preprocess)(const cv::Mat&);

            ArrayList<ArrayList<size_t>> SplitDatasetRandomly(int nFold);
            ArrayList<ArrayList<size_t>> SplitDatasetEqually(int nFold);

        private:
            TString datasetPath;
            TString configFilePath;

            System::XML::TiXmlDocument doc;
            ArrayList<TString> paths;
            ArrayList<int> labels;
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

            virtual void CrossValidation(int nFold = 3);
            ArrayList<double> GetPrecisions() { return precisions; }
            ArrayList<Word_f> GetWords() { return words; }
            ArrayList<Histogram> GetHistograms() { return histograms; }

        private:
            ArrayList<double> precisions;
            ArrayList<Word_f> words;
            ArrayList<Histogram> histograms;
        };

        template<typename LocalFeature>
        void LocalFeatureSolver<LocalFeature>::CrossValidation(int nFold)
        {
            srand(1);
            LoadData();
            ArrayList<TString> paths = GetPaths();
            ArrayList<int> labels = GetLabels();
            std::map<TString, TString> params = GetConfiguration(LocalFeature().GetName());
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
                Convert(machine(Preprocess(image)), features[i]);
            }

            ArrayList<ArrayList<size_t>> pass = SplitDatasetEqually(nFold);
            double maxPrecision = -1;
            this->precisions.Clear();
            for (int i = 0; i < nFold; i++)
            {
                printf("\nBegin Fold %d...\n", i + 1);
                ArrayList<size_t>& pickUpIndexes = pass[i];
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
                    Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels).first);

                if (this->precisions[i] > maxPrecision)
                {
                    maxPrecision = precisions[i];
                    this->words = words;
                    this->histograms = histograms;
                }

                printf("Fold %d Accuracy: %f\n", i + 1, this->precisions[i]);
            }

            printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(precisions), Math::StandardDeviation(precisions));
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

            virtual void CrossValidation(int nFold = 3);
            ArrayList<double> GetPrecisions() { return precisions; }
            ArrayList<GlobalFeatureVec_f> GetFeatures() { return features; }

        private:
            ArrayList<double> precisions;
            ArrayList<GlobalFeatureVec_f> features;
        };

        template<typename GlobalFeature>
        void GlobalFeatureSolver<GlobalFeature>::CrossValidation(int nFold)
        {
            srand(1);
            LoadData();
            ArrayList<TString> paths = GetPaths();
            ArrayList<int> labels = GetLabels();
            std::map<TString, TString> params = GetConfiguration(GlobalFeature().GetName());
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
                Convert(machine(Preprocess(image)), features[i]);
            }

            this->features = features;

            ArrayList<ArrayList<size_t>> pass = SplitDatasetEqually(nFold);
            double maxPrecision = -1;
            this->precisions.Clear();
            for (int i = 0; i < nFold; i++)
            {
                printf("\nBegin Fold %d...\n", i + 1);
                ArrayList<size_t>& pickUpIndexes = pass[i];
                ArrayList<GlobalFeatureVec_f> trainingSet = Divide(features, pickUpIndexes).Item2();
                ArrayList<GlobalFeatureVec_f> evaluationSet = Divide(features, pickUpIndexes).Item1();
                ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
                ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();

                this->precisions.Add(KNN<GlobalFeatureVec_f>().
                    Evaluate(trainingSet, trainingLabels, evaluationSet, evaluationLabels).first);

                printf("Fold %d Accuracy: %f\n", i + 1, this->precisions[i]);
            }

            printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(precisions), Math::StandardDeviation(precisions));
        }
    }
}
}