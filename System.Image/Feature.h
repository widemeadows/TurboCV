#pragma once

#include "Util.h"
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // APIs for pre-processing
        //////////////////////////////////////////////////////////////////////////

        cv::Mat Preprocess(const cv::Mat& sketchImage, bool thinning, cv::Size size);


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
            HOG() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "hog"; }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& filteredOrientImages, 
                const cv::Point& center, int cellSize, int cellNum);

        private:
            bool inited;
            LocalFeatureVec feature;
        };

        // Regularly Sampling HOG
        class RHOG : public LocalFeature
        {
        public:
            RHOG() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "rhog"; }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& filteredOrientImages, 
                const cv::Point& center, int blockSize, int cellNum);

        private:
            bool inited;
            LocalFeatureVec feature;
        };

        // Shape Based HOG
        class SHOG : public LocalFeature
        {
        public:
            SHOG() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "shog"; }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& orientChannels,
                const cv::Point& pivot, const ArrayList<cv::Point>& points, int cellNum);

        private:
            bool inited;
            LocalFeatureVec feature;
        };

        // Log-SHOG
        class LogSHOG : public LocalFeature
        {
        public:
            LogSHOG() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "lshog"; }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const ArrayList<cv::Mat>& orientChannels, 
                const cv::Point& pivot, int blockSize, int cellNum);

        private:
            bool inited;
            LocalFeatureVec feature;
        };

        // Histogram of Oriented Shape Context
        class HOOSC : public LocalFeature
        {
        public:
            HOOSC() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "hoosc"; }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Mat& orientImage, 
                const cv::Point& pivot, const ArrayList<cv::Point>& points,
                const ArrayList<double>& logDistances, int angleNum, int orientNum);

        private:
            bool inited;
            LocalFeatureVec feature;
        };

        // Regularly Sampling HOOSC
        class RHOOSC : public LocalFeature
        {
        public:
            RHOOSC() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "rhoosc"; }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Mat& orientImage, 
                const cv::Point& center, const ArrayList<cv::Point>& points, 
                const ArrayList<double>& logDistances, int angleNum, int orientNum);

        private:
            bool inited;
            LocalFeatureVec feature;
        };

        // Shape Context
        class SC : public LocalFeature
        {
        public:
            SC() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "sc"; };

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& pivot, const ArrayList<cv::Point>& pivots,
                const ArrayList<double>& logDistances, int angleNum);

        private:
            bool inited;
            LocalFeatureVec feature;
        };

        // Points Based Shape Context
        class PSC : public LocalFeature
        {
        public:
            PSC() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "psc"; }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& pivot, const ArrayList<cv::Point>& points,
                const ArrayList<double>& logDistances, int angleNum);

        private:
            bool inited;
            LocalFeatureVec feature;
        };

        // Regularly Sampling Shape Context
        class RSC : public LocalFeature
        {
        public:
            RSC() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "rsc"; }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& center, const ArrayList<cv::Point>& pivots, 
                const ArrayList<double>& logDistances, int angleNum);

        private:
            bool inited;
            LocalFeatureVec feature;
        };

        // Points Based Regularly Sampling Shape Context
        class RPSC : public LocalFeature
        {
        public:
            RPSC() : inited(false) {}

            virtual LocalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "rpsc"; }

        protected:
            LocalFeatureVec GetFeature(const cv::Mat& sketchImage);
            Descriptor GetDescriptor(const cv::Point& center, const ArrayList<cv::Point>& points, 
                const ArrayList<double>& logDistances, int angleNum);

        private:
            bool inited;
            LocalFeatureVec feature;
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
            GHOG() : inited(false) {}

            virtual GlobalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "ghog"; }

        protected:
            GlobalFeatureVec GetFeature(const cv::Mat& sketchImage);

        private:
            bool inited;
            GlobalFeatureVec feature;
        };

        // Model the Shape of Scene
        class GIST : public GlobalFeature
        {
        public:
            GIST() : inited(false) {}

            virtual GlobalFeatureVec operator()(const cv::Mat& sketchImage) 
            { 
                if (!inited)
                {
                    feature = GetFeature(sketchImage);
                    inited = true;
                }

                return feature;
            }

            virtual String GetName() const { return "gist"; }

        protected:
            GlobalFeatureVec GetFeature(const cv::Mat& sketchImage);
            ArrayList<cv::Mat> GetGaborsInFreqDomain(const cv::Size& size, 
                const ArrayList<int>& orientNumPerScale);

        private:
            bool inited;
            GlobalFeatureVec feature;
        };
    }
}
}