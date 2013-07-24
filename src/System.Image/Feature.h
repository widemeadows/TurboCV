#pragma once

#include "../System/System.h"
#include "Util.h"
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // APIs for Preprocessing
        //////////////////////////////////////////////////////////////////////////

        cv::Mat Preprocess(const cv::Mat& sketchImage, bool thinning, cv::Size size);


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
            const ArrayList<GlobalFeature_f>& features,
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
            cv::Mat distanceMatrix = Mat::zeros(nVec, nVec, CV_64F);
            
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
                const cv::Point& center, int cellSize, int cellNum);
        };

        // Regularly Sampling HOG
        class RHOG : public LocalFeature
        {
        public:
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
                const cv::Point& center, int blockSize, int cellNum);
        };

        // Shape Based HOG
        class SHOG : public LocalFeature
        {
        public:
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
                const cv::Point& pivot, const ArrayList<cv::Point>& points, int cellNum);
        };

        // Log-SHOG
        class LogSHOG : public LocalFeature
        {
        public:
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
                const cv::Point& pivot, int blockSize, int cellNum);
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