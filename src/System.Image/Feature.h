#pragma once

#include "../System/System.h"
#include "Core.h"
#include <cv.h>
#include <map>

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // APIs for Helper Functions
        //////////////////////////////////////////////////////////////////////////

        inline double GetDoubleValue(
            const std::map<String, String>& params, 
            const String& paramName, 
            const double defaultValue)
        {
            std::map<String, String>::const_iterator itr = params.find(paramName);

            if (itr == params.end())
                return defaultValue;
            else
                return Double::Parse(itr->second);
        }

        inline ArrayList<double> GetDoubleList(
            const std::map<String, String>& params, 
            const String& paramName, 
            const ArrayList<double>& defaultValue)
        {
            std::map<String, String>::const_iterator itr = params.find(paramName);

            if (itr == params.end())
                return defaultValue;
            else
            {
                ArrayList<String> tokens = itr->second.Split(",");
                ArrayList<double> values(tokens.Count());

                for (int i = tokens.Count() - 1; i >= 0; i--)
                    values[i] = Double::Parse(tokens[i]);

                return values;
            }
        }


        //////////////////////////////////////////////////////////////////////////
        // APIs for Save Features
        //////////////////////////////////////////////////////////////////////////

        void SaveLocalFeatures(
            const String& fileName,
            const ArrayList<Word_f>& words,
            const ArrayList<Histogram>& freqHists,
            const ArrayList<int>& labels);

        void SaveGlobalFeatures(
            const String& fileName, 
            const ArrayList<GlobalFeatureVec_f>& features,
            const ArrayList<int>& labels);

        template<typename T>
        cv::Mat SaveDistanceMatrix(
            const String& fileName, 
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
            
            FILE* file = fopen(fileName, "w");

            fprintf(file, "%d\n", nVec);

            for (int i = 0; i < nVec; i++)
            {
                fprintf(file, "%d", labels[i]);
                for (int j = 0; j < nVec; j++)
                    fprintf(file, " %f", distanceMatrix.at<double>(i, j));
                fprintf(file, "\n");
            }

            fclose(file);

            return distanceMatrix;
        }


        //////////////////////////////////////////////////////////////////////////
        // APIs for Local Features
        //////////////////////////////////////////////////////////////////////////

        class LocalFeature
        {
        public:
            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) = 0;
            virtual String GetName() const = 0;
        };

        // Histogram of Gradient
        class HOG : public LocalFeature
        {
        public:
            HOG(): orientNum(4), cellNum(4), cellSize(23) {}

            HOG(const std::map<String, String>& params)
            {
                orientNum = GetDoubleValue(params, "orientNum", 4);
                cellNum = GetDoubleValue(params, "cellNum", 4);
                cellSize = GetDoubleValue(params, "cellSize", 23);

                printf("OrientNum: %d, CellNum: %d, CellSize: %d\n", orientNum, cellNum, (int)cellSize);
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
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

            RHOG(const std::map<String, String>& params)
            {
                orientNum = GetDoubleValue(params, "orientNum", 4);
                cellNum = GetDoubleValue(params, "cellNum", 4);
                sampleNum = GetDoubleValue(params, "sampleNum", 28);
                blockSize = GetDoubleValue(params, "blockSize", 92);

                printf("OrientNum: %d, CellNum: %d, SampleNum: %d, BlockSize: %d\n", 
                    orientNum, cellNum, sampleNum, (int)blockSize);
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
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
            SHOG(): orientNum(4), cellNum(4) {}

            SHOG(const std::map<String, String>& params)
            {
                orientNum = GetDoubleValue(params, "orientNum", 4);
                cellNum = GetDoubleValue(params, "cellNum", 4);

                printf("OrientNum: %d, CellNum: %d\n", orientNum, cellNum);
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "shog"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& orientChannels,
                const cv::Point& pivot, const ArrayList<cv::Point>& points);

        private:
            int orientNum, cellNum;
        };

        // Log-SHOG
        class LogSHOG : public LocalFeature
        {
        public:
            LogSHOG(): orientNum(4), cellNum(4), scaleNum(15), sigmaInit(0.7), sigmaStep(1.2) {}

            LogSHOG(const std::map<String, String>& params)
            {
                orientNum = GetDoubleValue(params, "orientNum", 4);
                cellNum = GetDoubleValue(params, "cellNum", 4);
                scaleNum = GetDoubleValue(params, "scaleNum", 15);
                sigmaInit = GetDoubleValue(params, "sigmaInit", 0.7);
                sigmaStep = GetDoubleValue(params, "sigmaStep", 1.2);

                printf("OrientNum: %d, CellNum: %d, ScaleNum: %d, SigmaInit: %f, SigmaStep: %f\n", 
                    orientNum, cellNum, scaleNum, sigmaInit, sigmaStep);
            }

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "lshog"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& orientChannels, 
                const cv::Point& pivot, double blockSize);

        private:
            int orientNum, cellNum, scaleNum;
            double sigmaInit, sigmaStep;
        };

        // Histogram of Oriented Shape Context
        class HOOSC : public LocalFeature
        {
        public:
            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "hoosc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Mat& orientImage, 
                const cv::Point& pivot, const ArrayList<cv::Point>& points,
                const ArrayList<double>& logDistances, int angleNum, int orientNum);
        };

        // Regularly Sampling HOOSC
        class RHOOSC : public LocalFeature
        {
        public:
            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "rhoosc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Mat& orientImage, 
                const cv::Point& center, const ArrayList<cv::Point>& points, 
                const ArrayList<double>& logDistances, int angleNum, int orientNum);
        };

        // Shape Context
        class SC : public LocalFeature
        {
        public:
            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "sc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& pivot, const ArrayList<cv::Point>& pivots,
                const ArrayList<double>& logDistances, int angleNum);
        };

        // Points Based Shape Context
        class PSC : public LocalFeature
        {
        public:
            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "psc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& pivot, const ArrayList<cv::Point>& points,
                const ArrayList<double>& logDistances, int angleNum);
        };

        // Regularly Sampling Shape Context
        class RSC : public LocalFeature
        {
        public:
            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "rsc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& center, const ArrayList<cv::Point>& pivots, 
                const ArrayList<double>& logDistances, int angleNum);
        };

        // Points Based Regularly Sampling Shape Context
        class RPSC : public LocalFeature
        {
        public:
            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "rpsc"; 
            }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& center, const ArrayList<cv::Point>& points, 
                const ArrayList<double>& logDistances, int angleNum);
        };


        //////////////////////////////////////////////////////////////////////////
        // APIs for Global Features
        //////////////////////////////////////////////////////////////////////////

        class GlobalFeature
        {
        public:
            virtual GlobalFeatureVec operator()(const cv::Mat& sketchImage) = 0;
            virtual String GetName() const = 0;
        };

        // Global HOG
        class GHOG : public GlobalFeature
        {
        public:
            virtual GlobalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "ghog"; 
            }

        protected:
            GlobalFeatureVec GetFeature(const cv::Mat& sketchImage);
        };

        // Model the Shape of Scene
        class GIST : public GlobalFeature
        {
        public:
            virtual GlobalFeatureVec operator()(const cv::Mat& sketchImage) 
            {
                return GetFeature(sketchImage);
            }

            virtual String GetName() const 
            { 
                return "gist"; 
            }

        protected:
            GlobalFeatureVec GetFeature(const cv::Mat& sketchImage);
            ArrayList<cv::Mat> GetGaborsInFreqDomain(const cv::Size& size, 
                const ArrayList<int>& orientNumPerScale);
        };
    }
}
}