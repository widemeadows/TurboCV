#pragma once

#include "../System/System.h"
#include "../System.XML/System.XML.h"
#include <cv.h>

namespace TurboCV
{
namespace System
{
	namespace Image
	{
		//////////////////////////////////////////////////////////////////////////
		// Data Structures
		//////////////////////////////////////////////////////////////////////////

		typedef ArrayList<cv::Point> Edge;

		typedef ArrayList<double> Descriptor;
		typedef ArrayList<float> Descriptor_f;
		typedef ArrayList<Descriptor> LocalFeatureVec;
		typedef ArrayList<Descriptor_f> LocalFeatureVec_f;

		typedef ArrayList<double> GlobalFeatureVec;
		typedef ArrayList<float> GlobalFeatureVec_f;

        typedef ArrayList<cv::Point> PointList;
        typedef Group<ArrayList<PointList>, ArrayList<cv::Mat>> EdgeMatchInfo;

		typedef ArrayList<double> Word;
		typedef ArrayList<float> Word_f;
		typedef ArrayList<double> Histogram;
		typedef ArrayList<float> Histogram_f;

		const long long INF = std::numeric_limits<long long>::max();
        const double EPS = 1e-14;

		class PointHash
		{
        public:
			size_t operator()(const cv::Point& v) const
			{
				return v.y * 10000000 + v.x;
			}
		};

        // Two Dimensional Tree
        class TDTree
        {
        public:
            TDTree() {};
            TDTree(ArrayList<cv::Point>& points)
            {
                Build(points);
            }

            void Build(ArrayList<cv::Point>& points);
            Group<cv::Point, double> Find(const cv::Point& point);

        private:
            template<typename RandomAccessIterator>
            void innerBuild(const RandomAccessIterator& left, const RandomAccessIterator& right)
            {
                if (left >= right)
                    return;

                static auto compX = [](const cv::Point& u, const cv::Point& v) -> bool
                {
                    return u.x < v.x;
                };

                static auto compY = [](const cv::Point& u, const cv::Point& v) -> bool
                {
                    return u.y < v.y;
                };

                int minX = min_element(left, right, compX)->x;
                int maxX = max_element(left, right, compX)->x;
                int minY = min_element(left, right, compY)->y;
                int maxY = max_element(left, right, compY)->y;

                auto mid = left + (right - left) / 2;
                int midIdx = mid - nodes.begin();

                flags[midIdx] = (maxY - minY > maxX - minX);
                if (flags[midIdx])
                    nth_element(left, mid, right, compY);
                else
                    nth_element(left, mid, right, compX);

                innerBuild(left, mid);
                innerBuild(mid + 1, right);
            }

            template<typename RandomAccessIterator>
            void innerFind(
                const RandomAccessIterator& left, const RandomAccessIterator& right,
                const cv::Point& point)
            {
                if (left >= right)
                    return;

                static auto getSquareDistance = [](const cv::Point& u, const cv::Point& v) -> int
                {
                    return (u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y);
                };

                auto mid = left + (right - left) / 2;
                int midIdx = mid - nodes.begin();

                int distance = getSquareDistance(point, *mid);
                
                if (distance < minDist)
                {
                    minDist = distance;
                    nearest = *mid;
                }

                if (distance == 0)
                    return;
                
                int margin = flags[midIdx] ? (point.y - mid->y) : (point.x - mid->x);
                
                if (margin <= 0)
                    innerFind(left, mid, point);
                else
                    innerFind(mid + 1, right, point);

                if (margin * margin < minDist)
                {
                    if (margin <= 0)
                        innerFind(mid + 1, right, point);
                    else
                        innerFind(left, mid, point);
                }
            }

            ArrayList<cv::Point> nodes;
            ArrayList<int> flags;

            double minDist;
            cv::Point nearest;
        };


		//////////////////////////////////////////////////////////////////////////
		// APIs for Geometry Operations
		//////////////////////////////////////////////////////////////////////////

		double EulerDistance(const cv::Point& u, const cv::Point& v);
		ArrayList<double> EulerDistance(const cv::Point& u, const ArrayList<cv::Point>& vec);

		double Angle(const cv::Point& start, const cv::Point& end);
		ArrayList<double> Angle(const cv::Point& start, const ArrayList<cv::Point>& ends);


		//////////////////////////////////////////////////////////////////////////
		// APIs for Binary Images (Assume Background is 0)
		//////////////////////////////////////////////////////////////////////////

		void thin(cv::InputArray input, cv::OutputArray output, int iterations = 100);
		void clean(cv::InputArray input, cv::OutputArray output, int points = 1);

		ArrayList<Edge> EdgeLink(const cv::Mat& binaryImage, int minLength = 10);

        cv::Mat GetBoundingBox(const cv::Mat& binaryImage);
        ArrayList<cv::Point> GetEdgels(const cv::Mat& binaryImage);
        ArrayList<ArrayList<cv::Point>> GetEdgelChannels(const cv::Mat& binaryImage, int orientNum, 
            double sigma = 9, double lambda = 24);
        ArrayList<cv::Point> SampleOnShape(const cv::Mat& binaryImage, size_t samplingNum);

        cv::Mat EdgelsToMat(ArrayList<cv::Point> edgels);
        cv::Mat PerformGMMOnShape(const cv::Mat& binaryImage, size_t nComponent);


		//////////////////////////////////////////////////////////////////////////
		// APIs for Grayscale Images
		//////////////////////////////////////////////////////////////////////////

		cv::Mat reverse(cv::Mat image);
        cv::Mat FFTShift(const cv::Mat& image);
        void ConvolveFFT(cv::InputArray src, cv::OutputArray dst, int ddepth, cv::InputArray kernel);
        cv::Mat imshow(const cv::Mat& image, bool scale = true);

        Group<cv::Mat, cv::Mat> GetDiscreteGradient(const cv::Mat& image);

        Group<cv::Mat, cv::Mat> GetGaussDerivKernels(double sigma, double epsilon = 1e-2);
        ArrayList<cv::Mat> GetGaussDerivKernels(int orientNum, double sigma, double epsilon = 1e-2);
        ArrayList<cv::Mat> GetGaussDerivChannels(const cv::Mat& image, int orientNum, double sigma = 1.0);

        Group<cv::Mat, cv::Mat> GetGradient(const cv::Mat& image, double sigma = 1.0);
        ArrayList<cv::Mat> GetOrientChannels(const cv::Mat& image, int orientNum, double sigma = 1.0);

        cv::Mat GetStudentTKernel(cv::Size ksize, double sigma, double theta,
            double lambd, double gamma, double psi, int ktype = CV_64F);
        ArrayList<cv::Mat> GetGaborChannels(const cv::Mat& image, int orientNum, 
            double sigma = 4, double lambda = 10);

        cv::Mat GetLoGKernel(int ksize, double sigma, int ktype = CV_64F);
        ArrayList<cv::Mat> GetLoGPyramid(const cv::Mat& image, const ArrayList<double>& sigmas);
        ArrayList<cv::Mat> GetDoGPyramid(const cv::Mat& image, const ArrayList<double>& sigmas);

        ArrayList<cv::Point> SampleOnGrid(size_t height, size_t width, size_t numPerDirection);
        ArrayList<cv::Point> SampleFromPoints(const ArrayList<cv::Point>& points, size_t samplingNum);

        // Compute Gaussian and mean curvatures of a surface
        // return K(Gaussian curv), H(mean curv), Pmax(maximum curv), Pmin(minimum curv)
        ArrayList<cv::Mat> GetSurfaceCurvature(const cv::Mat& Z);
        ArrayList<cv::Mat> GetSurfaceCurvature(const cv::Mat& X, const cv::Mat& Y, const cv::Mat& Z);


		//////////////////////////////////////////////////////////////////////////
		// APIs for Vectors
		//////////////////////////////////////////////////////////////////////////

		template<typename RandomAccessIterator>
		inline void NormOneNormalize(const RandomAccessIterator& begin, const RandomAccessIterator& end)
		{
			double sum = 0;
			RandomAccessIterator curr = begin;

			do
			{
				sum += abs(*curr);     
			} while (++curr != end);

			if (sum != 0)
			{
				curr = begin;

				do
				{
					*curr /= sum;
				} while (++curr != end);
			}
		}

		template<typename RandomAccessIterator>
		inline void NormTwoNormalize(const RandomAccessIterator& begin, const RandomAccessIterator& end)
		{
			double sum = 0;
			RandomAccessIterator curr = begin;

			do
			{
				sum += (*curr) * (*curr);     
			} while (++curr != end);

			if (sum != 0)
			{
				double root = sqrt(sum);
				curr = begin;

				do
				{
					*curr /= root;
				} while (++curr != end);
			}
		}


        //////////////////////////////////////////////////////////////////////////
        // APIs for Configurations
        //////////////////////////////////////////////////////////////////////////

        std::map<TString, TString> LoadConfiguration(const TString& configFilePath, const TString& featureName);

        double GetDoubleValue(
            const std::map<TString, TString>& params, 
            const TString& paramName, 
            const double defaultValue);

        ArrayList<double> GetDoubleList(
            const std::map<TString, TString>& params, 
            const TString& paramName, 
            const ArrayList<double>& defaultValue);


        //////////////////////////////////////////////////////////////////////////
        // APIs for Datasets
        //////////////////////////////////////////////////////////////////////////

        Group<ArrayList<TString>, ArrayList<int>> LoadDataset(const TString& datasetPath);

        ArrayList<ArrayList<size_t>> SplitDatasetRandomly(const ArrayList<int> labels, int nFold);
        ArrayList<ArrayList<size_t>> SplitDatasetEqually(const ArrayList<int> labels, int nFold);


        //////////////////////////////////////////////////////////////////////////
        // APIs for Bag-of-Visual-Words
        //////////////////////////////////////////////////////////////////////////

        class BOV
        {
        public:
            BOV(const ArrayList<Descriptor_f>& descriptors, size_t clusterNum,
                size_t maxIter = 500, double epsilon = 1e-6)
            {
                this->descriptors = descriptors;
                this->clusterNum = clusterNum;
                this->termCriteria = cv::TermCriteria(CV_TERMCRIT_ITER, maxIter, epsilon);
            }

            ArrayList<Word_f> GetVisualWords()
            { 
                if (words.Count() != 0)
                    return words; 
                else
                {
                    auto result = GetVisualWords(descriptors, clusterNum, termCriteria);
                    words = result.Item1();
                    labels = result.Item2();

                    return words;
                }
            }

            ArrayList<size_t> GetWordLabels()
            {
                if (labels.Count() != 0)
                    return labels;
                else
                {
                    auto result = GetVisualWords(descriptors, clusterNum, termCriteria);
                    words = result.Item1();
                    labels = result.Item2();

                    return labels;
                }
            }

            ArrayList<double> GetSigmas()
            { 
                if (sigmas.Count() != 0)
                    return sigmas;
                else
                {
                    if (words.Count() == 0)
                    {
                        auto result = GetVisualWords(descriptors, clusterNum, termCriteria);
                        words = result.Item1();
                        labels = result.Item2();
                    }

                    return sigmas = GetVarianceSigmas(descriptors, labels, words);
                }
            }

        protected:
            static Group<ArrayList<Word_f>, ArrayList<size_t>> GetVisualWords(
                const ArrayList<Descriptor_f>& descriptors, 
                size_t clusterNum, 
                cv::TermCriteria termCriteria);

            static ArrayList<double> GetEntropySigmas(
                const ArrayList<Descriptor_f>& descriptors,
                const ArrayList<Word_f>& words,
                double perplexity = 60);

            static ArrayList<double> GetVarianceSigmas(
                const ArrayList<Descriptor_f>& descriptors,
                const ArrayList<size_t>& labels,
                const ArrayList<Word_f>& words);

        private:
            ArrayList<Descriptor_f> descriptors;
            size_t clusterNum;
            cv::TermCriteria termCriteria;

            ArrayList<Word_f> words;
            ArrayList<size_t> labels;
            ArrayList<double> sigmas;
        };


        class FreqHist
        {
        public:
            FreqHist(const ArrayList<LocalFeatureVec_f>& features, 
                     const ArrayList<Word_f>& words,
                     const ArrayList<double>& sigmas = ArrayList<double>())
            {
                this->features = features;
                this->words = words;

                if (sigmas.Count() != words.Count())
                {
                    ArrayList<double> initSigmas(words.Count());
                    for (int i = words.Count() - 1; i >= 0; i--)
                        initSigmas[i] = 0.1;

                    this->sigmas = initSigmas;
                }
                else
                    this->sigmas = sigmas;
            }

            ArrayList<double> GetSigmas()
            {
                return sigmas;
            }

            ArrayList<Histogram> GetFrequencyHistograms()
            {
                if (histograms.Count() != 0)
                    return histograms;
                else
                    return histograms = ComputeFrequencyHistograms(); 
            }

            void ReleaseFrequencyHistograms()
            {
                histograms = ArrayList<Histogram>();
            }
            
            ArrayList<LocalFeatureVec> GetPoolingHistograms(int nPool)
            {
                if (pools.Count() != 0)
                    return pools;
                else
                    return pools = ComputePoolingHistograms(nPool);
            }

            void ReleasePoolingHistograms()
            {
                pools = ArrayList<LocalFeatureVec>();
            }

            ArrayList<LocalFeatureVec_f> GetReconstructedInputs()
            {
                if (reconstructeds.Count() != 0)
                    return reconstructeds;
                else
                    return reconstructeds = ComputeReconstructedInputs();
            }

            void ReleaseReconstructedInputs()
            {
                reconstructeds = ArrayList<LocalFeatureVec_f>();
            }

            ArrayList<LocalFeatureVec_f> GetReconstructErrors()
            {
                if (errors.Count() != 0)
                    return errors;
                else
                    return errors = ComputeReconstructErrors();
            }

            void ReleaseReconstructErrors()
            {
                errors = ArrayList<LocalFeatureVec_f>();
            }

        protected:
            ArrayList<Histogram> ComputeFrequencyHistograms();
            Histogram ComputeFrequencyHistogram(const LocalFeatureVec_f& feature);

            ArrayList<LocalFeatureVec> ComputePoolingHistograms(int nPool);
            LocalFeatureVec ComputePoolingHistogram(const LocalFeatureVec_f& feature, int nPool);

            ArrayList<LocalFeatureVec_f> ComputeReconstructedInputs();
            LocalFeatureVec_f ComputeReconstructedInput(const LocalFeatureVec_f& feature);

            ArrayList<LocalFeatureVec_f> ComputeReconstructErrors();
            LocalFeatureVec_f ComputeReconstructError(const LocalFeatureVec_f& feature);

            ArrayList<double> GetDistancesToVisualWords(const Descriptor_f& descriptor);

        private:
            ArrayList<LocalFeatureVec_f> features;
            ArrayList<Word_f> words;
            ArrayList<double> sigmas;

            ArrayList<Histogram> histograms;
            ArrayList<LocalFeatureVec> pools;
            ArrayList<LocalFeatureVec_f> reconstructeds;
            ArrayList<LocalFeatureVec_f> errors;
        };

        inline ArrayList<Descriptor_f> SampleDescriptors(const ArrayList<Descriptor_f>& descriptors,
            size_t sampleNum = INF)
        {
            sampleNum = cv::min(descriptors.Count(), sampleNum);
            return RandomPickUp(descriptors, sampleNum);
        }

        inline ArrayList<Descriptor_f> SampleDescriptors(const ArrayList<LocalFeatureVec_f>& features, 
            size_t sampleNum = INF)
        {
            ArrayList<Descriptor_f> allDescriptors;
            for (size_t i = 0; i < features.Count(); i++)
                for (size_t j = 0; j < features[i].Count(); j++)
                    allDescriptors.Add(features[i][j]);

            sampleNum = cv::min(allDescriptors.Count(), sampleNum);
            return RandomPickUp(allDescriptors, sampleNum);
        }

        inline ArrayList<Histogram> PerformBOV(const ArrayList<LocalFeatureVec_f>& features,
            size_t clusterNum, size_t sampleNum)
        {
            ArrayList<Descriptor_f> samples = SampleDescriptors(features, sampleNum);
            ArrayList<Word_f> words = BOV(samples, clusterNum).GetVisualWords();
            return FreqHist(features, words).GetFrequencyHistograms();
        }


		//////////////////////////////////////////////////////////////////////////
		// Others
		//////////////////////////////////////////////////////////////////////////

		inline void Convert(const LocalFeatureVec& src, LocalFeatureVec_f& dst)
		{
			dst.Clear();

			for (auto descriptor : src)
			{
				Descriptor_f tmp(descriptor.Count());
				for (int i = 0; i < descriptor.Count(); i++)
					tmp[i] = (float)descriptor[i];

				dst.Add(tmp);
			}

			dst.Shrink();
		}

		inline void Convert(const GlobalFeatureVec& src, GlobalFeatureVec_f& dst)
		{
			dst.Clear();

			for (auto item : src)
				dst.Add((float)item);

			dst.Shrink();
		}

        inline size_t RoundIndex(int index, size_t period, bool cyclic = false)
        {
            if (cyclic)
            {
                while (index < 0)
                    index += period;

                return (size_t)index % period;
            }
            else
            {
                if (index < 0)
                    return 0;
                else if (index >= period)
                    return period - 1;
                else
                    return index;
            }
        }

        inline size_t FindBinIndex(double value, double minIncluded, double maxExcluded, 
            size_t intervalNum, bool cyclic = false)
        {
            assert(intervalNum > 0 && maxExcluded > minIncluded);
            double intervalSize = (maxExcluded - minIncluded) / intervalNum;

            int index = (int)(value / intervalSize);
            index = RoundIndex(index, intervalNum, cyclic);
            assert(index >= 0 && index < intervalNum);

            return index;
        }

        inline ArrayList<double> linspace(double start, double end, int pointNum)
        {
            double size = (end - start) / (pointNum - 1);

            ArrayList<double> result;
            result.Add(start);
            for (int i = 1; i < pointNum - 1; i++)
                result.Add(result[i - 1] + size);
            result.Add(end);

            assert(result.Count() == pointNum);
            return result;
        }

        // Params:
        // 1. distances -- Distances from database images to the query;
        // 2. relevants -- If image[i] and the query belong to the same category, 
        //                 then relevants[i] is true;
        // 3. numOfCP -- Number of Control Points.
        inline Group<ArrayList<double>, ArrayList<double>> roc(
            const ArrayList<double>& distances, 
            const ArrayList<bool>& relevants,
            int numOfCP = 20)
        {
            ArrayList<double> positiveDist, negativeDist;
            for (int i = 0; i < relevants.Count(); i++)
            {
                if (relevants[i])
                    positiveDist.Add(distances[i]);
                else
                    negativeDist.Add(distances[i]);
            }

            double firstCP = Math::Min(distances);
            double lastCP = Math::Max(distances);
            ArrayList<double> plot = linspace(firstCP, lastCP, numOfCP);

            ArrayList<double> TP(numOfCP), FP(numOfCP), TN(numOfCP), FN(numOfCP);
            for (int i = 0; i < numOfCP; i++)
            {
                for (auto item : positiveDist)
                    if (item <= plot[i])
                        TP[i]++;

                for (auto item : negativeDist)
                    if (item <= plot[i])
                        FP[i]++;

                for (auto item : positiveDist)
                    if (item > plot[i])
                        FN[i]++;

                for (auto item : negativeDist)
                    if (item > plot[i])
                        TN[i]++;

                assert(TP[i] + FN[i] == positiveDist.Count() && 
                    FP[i] + TN[i] == negativeDist.Count());
            }

            ArrayList<double> DR, FPR;
            for (int i = 0; i < numOfCP; i++)
            {
                DR.Add(TP[i] / (TP[i] + FN[i]));
                FPR.Add(FP[i] / (FP[i] + TN[i]));
            }

            return CreateGroup(DR, FPR);
        }
	}
}
}
