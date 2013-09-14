#pragma once

#include "../System/System.h"
#include "Core.h"
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // APIs for Save Features
        //////////////////////////////////////////////////////////////////////////

        void SaveLocalFeatures(
            const TString& fileName,
            const ArrayList<Word_f>& words,
            const ArrayList<Histogram>& freqHists,
            const ArrayList<int>& labels);

        void SaveGlobalFeatures(
            const TString& fileName, 
            const ArrayList<GlobalFeatureVec_f>& features,
            const ArrayList<int>& labels);

        template<typename T>
        cv::Mat SaveDistanceMatrix(
            const TString& fileName, 
            const ArrayList<T>& vecs, 
            const ArrayList<int>& labels, 
            double (*getDistance)(const T&, const T&) = Math::NormOneDistance)
        {
            if (vecs.Count() == 0)
                return cv::Mat();

            int nVec = vecs.Count();
            cv::Mat distanceMatrix(nVec, nVec, CV_64F);
            
            #pragma omp parallel for
            for (int i = 0; i < nVec; i++)
                for (int j = 0; j < nVec; j++)
                    distanceMatrix.at<double>(i, j) = getDistance(vecs[i], vecs[j]);
            
            SaveDistanceMatrix(fileName, distanceMatrix, labels);
            return distanceMatrix;
        }

        void SaveDistanceMatrix(
            const TString& fileName,
            const cv::Mat& distanceMatrix,
            const ArrayList<int>& labels
            );


        //////////////////////////////////////////////////////////////////////////
        // APIs for Local Features
        //////////////////////////////////////////////////////////////////////////

        class LocalFeature
        {
        public:
            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) = 0;
            virtual TString GetName() const = 0;
        };

        // Histogram of Gradient
        class HOG : public LocalFeature
        {
        public:
            HOG(): orientNum(4), cellNum(6), cellSize(14) {}

            HOG(const std::map<TString, TString>& params, bool printParams = false)
            {
                orientNum = GetDoubleValue(params, "orientNum", 4);
                cellNum = GetDoubleValue(params, "cellNum", 6);
                cellSize = GetDoubleValue(params, "cellSize", 14);

                if (printParams)
                {
                    printf("OrientNum: %d, CellNum: %d, CellSize: %d\n", orientNum, cellNum, (int)cellSize);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "hog"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& filteredOrientImages, 
                const cv::Point& center);

        private:
            int orientNum, cellNum;
            double cellSize;
        };

        // Regularly Sampling HOG
        class RHOG : public LocalFeature
        {
        public:
            RHOG(): orientNum(4), cellNum(4), sampleNum(28), blockSize(92) {}

            RHOG(const std::map<TString, TString>& params, bool printParams = false)
            {
                orientNum = GetDoubleValue(params, "orientNum", 4);
                cellNum = GetDoubleValue(params, "cellNum", 4);
                sampleNum = GetDoubleValue(params, "sampleNum", 28);
                blockSize = GetDoubleValue(params, "blockSize", 92);

                if (printParams)
                {
                    printf("OrientNum: %d, CellNum: %d, SampleNum: %d, BlockSize: %d\n", 
                        orientNum, cellNum, sampleNum, (int)blockSize);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "rhog"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& filteredOrientImages, 
                const cv::Point& center);

        private:
            int orientNum, cellNum, sampleNum;
            double blockSize;
        };

        // Shape Based HOG
        class SHOG : public LocalFeature
        {
        public:
            SHOG(): orientNum(4), cellNum(4), pivotRatio(0.33) {}

            SHOG(const std::map<TString, TString>& params, bool printParams = false)
            {
                orientNum = GetDoubleValue(params, "orientNum", 4);
                cellNum = GetDoubleValue(params, "cellNum", 4);
                pivotRatio = GetDoubleValue(params, "pivotRatio", 0.33);

                if (printParams)
                {
                    printf("OrientNum: %d, CellNum: %d, PivotRatio: %f\n", orientNum, cellNum, pivotRatio);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "shog"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& orientChannels,
                const cv::Point& pivot, const ArrayList<cv::Point>& points);

        private:
            int orientNum, cellNum;
            double pivotRatio;
        };

        // Log-SHOG
        class LogSHOG : public LocalFeature
        {
        public:
            LogSHOG(): orientNum(4), cellNum(4), scaleNum(15), sigmaInit(0.7), 
                sigmaStep(1.2), pivotRatio(0.33) {}

            LogSHOG(const std::map<TString, TString>& params, bool printParams = false)
            {
                orientNum = GetDoubleValue(params, "orientNum", 4);
                cellNum = GetDoubleValue(params, "cellNum", 4);
                scaleNum = GetDoubleValue(params, "scaleNum", 15);
                sigmaInit = GetDoubleValue(params, "sigmaInit", 0.7);
                sigmaStep = GetDoubleValue(params, "sigmaStep", 1.2);
                pivotRatio = GetDoubleValue(params, "pivotRatio", 0.33);

                if (printParams)
                {
                    printf("OrientNum: %d, CellNum: %d, ScaleNum: %d, SigmaInit: %f, SigmaStep: %f, PivotRatio: %f\n", 
                        orientNum, cellNum, scaleNum, sigmaInit, sigmaStep, pivotRatio);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "lshog"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& orientChannels, 
                const cv::Point& pivot, double blockSize);

        private:
            int orientNum, cellNum, scaleNum;
            double sigmaInit, sigmaStep, pivotRatio;
        };

        // Histogram of Oriented Shape Context
        class HOOSC : public LocalFeature
        {
        public:
            HOOSC(): angleNum(6), orientNum(6), pivotRatio(0.33)
            {
                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double));
            }

            HOOSC(const std::map<TString, TString>& params, bool printParams = false)
            {
                angleNum = GetDoubleValue(params, "angleNum", 6);
                orientNum = GetDoubleValue(params, "orientNum", 6);
                pivotRatio = GetDoubleValue(params, "pivotRatio", 0.33);

                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = GetDoubleList(params, "logDistances", 
                    ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double)));

                if (printParams)
                {
                    printf("LogDistances:");
                    for (int i = 0; i < logDistances.Count(); i++)
                        printf(" %f", logDistances[i]);
                    printf("\n");

                    printf("AngleNum: %d, OrientNum: %d, PivotRatio: %f\n", angleNum, orientNum, pivotRatio);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "hoosc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Mat& orientImage, const cv::Point& pivot, 
                const ArrayList<cv::Point>& points);

        private:
            ArrayList<double> logDistances;
            int angleNum, orientNum;
            double pivotRatio;
        };

        // Regularly Sampling HOOSC
        class RHOOSC : public LocalFeature
        {
        public:
            RHOOSC(): angleNum(9), orientNum(8), sampleNum(28)
            {
                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double));
            }

            RHOOSC(const std::map<TString, TString>& params, bool printParams = false)
            {
                angleNum = GetDoubleValue(params, "angleNum", 9);
                orientNum = GetDoubleValue(params, "orientNum", 8);
                sampleNum = GetDoubleValue(params, "sampleNum", 28);

                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = GetDoubleList(params, "logDistances", 
                    ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double)));

                if (printParams)
                {
                    printf("LogDistances:");
                    for (int i = 0; i < logDistances.Count(); i++)
                        printf(" %f", logDistances[i]);
                    printf("\n");

                    printf("AngleNum: %d, OrientNum: %d, SampleNum: %f\n", angleNum, orientNum, sampleNum);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "rhoosc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Mat& orientImage, const cv::Point& center, 
                const ArrayList<cv::Point>& points);

        private:
            ArrayList<double> logDistances;
            int angleNum, orientNum, sampleNum;
        };

        // Shape Context
        class SC : public LocalFeature
        {
        public:
            SC(): angleNum(12), pivotRatio(0.33)
            {
                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double));
            }

            SC(const std::map<TString, TString>& params, bool printParams = false)
            {
                angleNum = GetDoubleValue(params, "angleNum", 12);
                pivotRatio = GetDoubleValue(params, "pivotRatio", 0.33);

                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = GetDoubleList(params, "logDistances", 
                    ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double)));

                if (printParams)
                {
                    printf("LogDistances:");
                    for (int i = 0; i < logDistances.Count(); i++)
                        printf(" %f", logDistances[i]);
                    printf("\n");

                    printf("AngleNum: %d, PivotRatio: %f\n", angleNum, pivotRatio);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "sc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& pivot, const ArrayList<cv::Point>& pivots);

        private:
            ArrayList<double> logDistances;
            int angleNum;
            double pivotRatio;
        };

        // Points Based Shape Context
        class PSC : public LocalFeature
        {
        public:
            PSC(): angleNum(12), pivotRatio(0.33)
            {
                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double));
            }

            PSC(const std::map<TString, TString>& params, bool printParams = false)
            {
                angleNum = GetDoubleValue(params, "angleNum", 12);
                pivotRatio = GetDoubleValue(params, "pivotRatio", 0.33);

                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = GetDoubleList(params, "logDistances", 
                    ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double)));

                if (printParams)
                {
                    printf("LogDistances:");
                    for (int i = 0; i < logDistances.Count(); i++)
                        printf(" %f", logDistances[i]);
                    printf("\n");

                    printf("AngleNum: %d, PivotRatio: %f\n", angleNum, pivotRatio);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "psc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& pivot, const ArrayList<cv::Point>& points);

        private:
            ArrayList<double> logDistances;
            int angleNum;
            double pivotRatio;
        };

        // Regularly Sampling Shape Context
        class RSC : public LocalFeature
        {
        public:
            RSC(): angleNum(12), sampleNum(28), pivotRatio(0.33)
            {
                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double));
            }

            RSC(const std::map<TString, TString>& params, bool printParams = false)
            {
                angleNum = GetDoubleValue(params, "angleNum", 12);
                sampleNum = GetDoubleValue(params, "sampleNum", 28);
                pivotRatio = GetDoubleValue(params, "pivotRatio", 0.33);

                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = GetDoubleList(params, "logDistances", 
                    ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double)));

                if (printParams)
                {
                    printf("LogDistances:");
                    for (int i = 0; i < logDistances.Count(); i++)
                        printf(" %f", logDistances[i]);
                    printf("\n");

                    printf("AngleNum: %d, SampleNum: %d, PivotRatio: %f\n", angleNum, sampleNum, pivotRatio);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "rsc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& center, const ArrayList<cv::Point>& pivots);

        private:
            ArrayList<double> logDistances;
            int angleNum, sampleNum;
            double pivotRatio;
        };

        // Points Based Regularly Sampling Shape Context
        class RPSC : public LocalFeature
        {
        public:
            RPSC(): angleNum(12), sampleNum(28)
            {
                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double));
            }

            RPSC(const std::map<TString, TString>& params, bool printParams = false)
            {
                angleNum = GetDoubleValue(params, "angleNum", 12);
                sampleNum = GetDoubleValue(params, "sampleNum", 28);

                double tmp[] = {0, 0.125, 0.25, 0.5, 1, 2};
                logDistances = GetDoubleList(params, "logDistances", 
                    ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double)));

                if (printParams)
                {
                    printf("LogDistances:");
                    for (int i = 0; i < logDistances.Count(); i++)
                        printf(" %f", logDistances[i]);
                    printf("\n");

                    printf("AngleNum: %d, SampleNum: %d\n", angleNum, sampleNum);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "rpsc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& center, const ArrayList<cv::Point>& points, 
                const ArrayList<double>& logDistances, int angleNum);

        private:
            ArrayList<double> logDistances;
            int angleNum, sampleNum;
        };

        // Regularly Sampling Gabor
        class RGabor : public LocalFeature
        {
        public:
            RGabor(): orientNum(9), cellNum(4), sampleNum(28), blockSize(92) {}

            RGabor(const std::map<TString, TString>& params, bool printParams = false)
            {
                orientNum = GetDoubleValue(params, "orientNum", 9);
                cellNum = GetDoubleValue(params, "cellNum", 4);
                sampleNum = GetDoubleValue(params, "sampleNum", 28);
                blockSize = GetDoubleValue(params, "blockSize", 92);

                if (printParams)
                {
                    printf("OrientNum: %d, CellNum: %d, SampleNum: %d, BlockSize: %d\n", 
                        orientNum, cellNum, sampleNum, (int)blockSize);
                }
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "rgabor"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);

        private:
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& filteredOrientImages, 
                const cv::Point& center);

            int orientNum, cellNum, sampleNum;
            double blockSize;
        };

        //////////////////////////////////////////////////////////////////////////
        // APIs for Global Features
        //////////////////////////////////////////////////////////////////////////

        class GlobalFeature
        {
        public:
            virtual GlobalFeatureVec operator()(const cv::Mat& sketchImage) = 0;
            virtual TString GetName() const = 0;
        };

        // Global HOG
        class GHOG : public GlobalFeature
        {
        public:
            GHOG(): orientNum(8), blockSize(48) {}

            GHOG(const std::map<TString, TString>& params, bool printParams = false)
            {
                orientNum = GetDoubleValue(params, "orientNum", 8);
                blockSize = GetDoubleValue(params, "blockSize", 48);

                if (printParams)
                {
                    printf("OrientNum: %d, BlockSize: %d\n", orientNum, blockSize);
                }
            }

            virtual GlobalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "ghog"; 
            }

        protected:
            GlobalFeatureVec GetFeature(const cv::Mat& sketchImage);

        private:
            int orientNum, blockSize;
        };

        // Model the Shape of Scene
        class GIST : public GlobalFeature
        {
        public:
            GIST(): blockNum(4)
            {
                int tmp[] = {8, 8, 8, 8};
                orientNumPerScale = ArrayList<int>(tmp, tmp + sizeof(tmp) / sizeof(int));
            }

            GIST(const std::map<TString, TString>& params, bool printParams = false)
            {
                blockNum = GetDoubleValue(params, "blockNum", 4);

                double tmp[] = {8, 8, 8, 8};
                ArrayList<double> orientNumList = GetDoubleList(params, "orientNumPerScale", 
                    ArrayList<double>(tmp, tmp + sizeof(tmp) / sizeof(double)));
                for (int i = 0; i < orientNumList.Count(); i++)
                    orientNumPerScale.Add(orientNumList[i]);

                if (printParams)
                {
                    printf("OrientNumPerScale:");
                    for (int i = 0; i < orientNumPerScale.Count(); i++)
                        printf(" %d", orientNumPerScale[i]);
                    printf("\n");

                    printf("BlockNum: %d\n", blockNum);
                }
            }

            virtual GlobalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual TString GetName() const 
            { 
                return "gist"; 
            }

        protected:
            GlobalFeatureVec GetFeature(const cv::Mat& sketchImage);
            ArrayList<cv::Mat> GetGaborsInFreqDomain(const cv::Size& size);

        private:
            int blockNum;
            ArrayList<int> orientNumPerScale;
        };


        //////////////////////////////////////////////////////////////////////////
        // APIs for EdgeMatch
        //////////////////////////////////////////////////////////////////////////

        class EdgeMatch
        {
        public:
            EdgeMatchInfo GetEdgeMatchInfo(const cv::Mat& sketchImage)
            {
                ArrayList<PointList> channels = GetChannels(sketchImage);
                ArrayList<cv::Mat> transforms = GetTransforms(sketchImage);

                return CreateGroup(channels, transforms);
            }

            double GetTwoWayDistance(const EdgeMatchInfo& uInfo, const EdgeMatchInfo& vInfo)
            {
                double uToV = GetOneWayDistance(uInfo.Item1(), vInfo.Item2());
                double vToU = GetOneWayDistance(vInfo.Item1(), uInfo.Item2());

                return GetTwoWayDistance(uToV, vToU);
            }

            ArrayList<cv::Mat> GetTransforms(const cv::Mat& sketchImage)
            {
                return GetTransforms(sketchImage.size(), GetChannels(sketchImage));
            }

            virtual ArrayList<PointList> GetChannels(const cv::Mat& sketchImage) = 0;
            virtual ArrayList<cv::Mat> GetTransforms(const cv::Size& size, const ArrayList<PointList> channels) = 0;

            virtual double GetOneWayDistance(const ArrayList<PointList>& u, const ArrayList<cv::Mat>& v) = 0;
            virtual double GetTwoWayDistance(double uToV, double vToU) = 0;

            virtual TString GetName() const = 0;
        };

        // Chamfer Matching
        class CM : public EdgeMatch
        {
        public:
            CM(): maxDistance(40) {}

            CM(const std::map<TString, TString>& params, bool printParams = false)
            {
                maxDistance = GetDoubleValue(params, "maxDistance", 40);

                if (printParams)
                {
                    printf("MaxDistance: %f\n", maxDistance);
                }
            }

            virtual ArrayList<PointList> GetChannels(const cv::Mat& sketchImage);
            virtual ArrayList<cv::Mat> GetTransforms(const cv::Size& size, const ArrayList<PointList> channels);

            virtual double GetOneWayDistance(const ArrayList<PointList>& u, const ArrayList<cv::Mat>& v);
            virtual double GetTwoWayDistance(double uToV, double vToU);

            virtual TString GetName() const 
            { 
                return "cm"; 
            }

        private:
            double maxDistance;
        };

        // Oriented Chamfer Matching
        class OCM : public EdgeMatch
        {
        public:
            OCM(): orientNum(6), maxDistance(40) {}

            OCM(const std::map<TString, TString>& params, bool printParams = false)
            {
                orientNum = GetDoubleValue(params, "orientNum", 6);
                maxDistance = GetDoubleValue(params, "maxDistance", 40);

                if (printParams)
                {
                    printf("OrientNum: %d, MaxDistance: %f\n", orientNum, maxDistance);
                }
            }

            virtual ArrayList<PointList> GetChannels(const cv::Mat& sketchImage);
            virtual ArrayList<cv::Mat> GetTransforms(const cv::Size& size, const ArrayList<PointList> channels);

            virtual double GetOneWayDistance(const ArrayList<PointList>& u, const ArrayList<cv::Mat>& v);
            virtual double GetTwoWayDistance(double uToV, double vToU);

            virtual TString GetName() const 
            { 
                return "ocm"; 
            }

        private:
            int orientNum;
            double maxDistance;
        };

        // Hitmap
        class Hitmap : public EdgeMatch
        {
        public:
            Hitmap(): orientNum(6), maxDistance(22) {}

            Hitmap(const std::map<TString, TString>& params, bool printParams = false)
            {
                orientNum = GetDoubleValue(params, "orientNum", 6);
                maxDistance = GetDoubleValue(params, "maxDistance", 22);

                if (printParams)
                {
                    printf("OrientNum: %d, MaxDistance: %f\n", orientNum, maxDistance);
                }
            }

            virtual ArrayList<PointList> GetChannels(const cv::Mat& sketchImage);
            virtual ArrayList<cv::Mat> GetTransforms(const cv::Size& size, const ArrayList<PointList> channels);

            virtual double GetOneWayDistance(const ArrayList<PointList>& u, const ArrayList<cv::Mat>& v);
            virtual double GetTwoWayDistance(double uToV, double vToU);

            virtual TString GetName() const 
            { 
                return "hit"; 
            }

        private:
            int orientNum;
            double maxDistance;
        };   
    }
}
}