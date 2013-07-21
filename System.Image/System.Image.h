#pragma once

#include "../System/Collection.h"
#include <cv.h>

namespace TurboCV
{
namespace System
{
	namespace Image
	{
		//////////////////////////////////////////////////////////////////////////
		// Data Structure
		//////////////////////////////////////////////////////////////////////////

		typedef ArrayList<cv::Point> Edge;

		typedef ArrayList<double> Descriptor;
		typedef ArrayList<float> Descriptor_f;
		typedef ArrayList<Descriptor> LocalFeatureVec;
		typedef ArrayList<Descriptor_f> LocalFeature_f;
		typedef ArrayList<double> GlobalFeatureVec;
		typedef ArrayList<float> GlobalFeature_f;

		typedef ArrayList<double> Word;
		typedef ArrayList<float> Word_f;
		typedef ArrayList<double> Histogram;
		typedef ArrayList<float> Histogram_f;

		const long long INF = std::numeric_limits<long long>::max();
		const double EPS = 1e-14;
		const int MAX_GRAYSCALE = 255;

		class PointComp
		{
			bool operator()(const cv::Point& u, const cv::Point& v) const
			{
				if (u.x < v.x)
					return true;
				else if (u.x == v.x)
					return u.y < v.y;
				else
					return false;
			}
		};

		class PointHash
		{
			size_t operator()(const cv::Point& v) const
			{
				return v.y * 10000000 + v.x;
			}
		};


		//////////////////////////////////////////////////////////////////////////
		// APIs for Geometry Operations
		//////////////////////////////////////////////////////////////////////////

		double EulerDistance(const Point& u, const Point& v);
		ArrayList<double> EulerDistance(const Point& u, const ArrayList<Point>& vec);

		double Angle(const Point& start, const Point& end);
		ArrayList<double> Angle(const Point& start, const ArrayList<Point>& ends);


		//////////////////////////////////////////////////////////////////////////
		// APIs for Binary Images (Edgels are 1 and Background is 0)
		//////////////////////////////////////////////////////////////////////////

		void thin(cv::InputArray input, cv::OutputArray output, int iterations = 100);
		void clean(cv::InputArray input, cv::OutputArray output, int points = 1);

		ArrayList<Edge> EdgeLink(const Mat& binaryImage, int minLength = 10);

        ArrayList<cv::Point> GetEdgels(const cv::Mat& binaryImage);
        ArrayList<Point> SampleOnShape(const Mat& binaryImage, size_t samplingNum);


		//////////////////////////////////////////////////////////////////////////
		// APIs for Grayscale Images
		//////////////////////////////////////////////////////////////////////////

		cv::Mat reverse(cv::Mat image);
        cv::Mat FFTShift(const cv::Mat& image);
        void ConvolveDFT(cv::InputArray src, cv::OutputArray dst, int ddepth, cv::InputArray kernel);
        cv::Mat imshow(const cv::Mat& image, bool scale = true);

        Group<cv::Mat, cv::Mat> GetGradientKernel(double sigma, double epsilon);
        Group<cv::Mat, cv::Mat> GetGradient(const cv::Mat& image, double sigma = 1.0);
        ArrayList<cv::Mat> GetOrientChannels(const cv::Mat& image, int orientNum);

        cv::Mat GetLoGKernel(int ksize, double sigma, int ktype = CV_64F);
        ArrayList<cv::Mat> GetLoGPyramid(const cv::Mat& image, const ArrayList<double>& sigmas);
        ArrayList<cv::Mat> GetDoGPyramid(const cv::Mat& image, const ArrayList<double>& sigmas);

        ArrayList<cv::Point> SampleOnGrid(size_t height, size_t width, size_t numPerDirection);
        ArrayList<cv::Point> SampleFromPoints(const ArrayList<cv::Point>& points, size_t samplingNum);


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
		// Others
		//////////////////////////////////////////////////////////////////////////

		inline void Convert(const LocalFeatureVec& src, LocalFeature_f& dst)
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

		inline void Convert(const GlobalFeatureVec& src, GlobalFeature_f& dst)
		{
			dst.Clear();

			for (auto item : src)
				dst.Add((float)item);

			dst.Shrink();
		}
	}
}
}
