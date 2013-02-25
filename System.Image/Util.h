#pragma once

#include <cv.h>
using namespace cv;

namespace System
{
    namespace Image
    {
		typedef vector<double> Histogram;
		typedef vector<double> Descriptor;
		typedef vector<Descriptor> Feature;

		const int INF = 2147483647;
        const int MAX_GRAYSCALE = 255;

        inline Mat reverse(Mat grayScaleImage) 
        {
            assert(grayScaleImage.type() == CV_8U);
            Mat result(grayScaleImage.size(), CV_8U);

            for (int i = 0; i < grayScaleImage.rows; i++)
                for (int j = 0; j < grayScaleImage.cols; j++)
                    result.at<uchar>(i, j) = MAX_GRAYSCALE - grayScaleImage.at<uchar>(i, j);

            return result;
        }

		template<typename T>
		inline T min(T u, T v)
		{
			return u < v ? u : v;
		}

		template<typename T>
        inline void NormOneNormalize(vector<T>& vec)
        {
            double sum = 0;

            for (T item : vec)
                sum += item;

            if (sum != 0)
		        for (int i = vec.size() - 1; i >= 0; i--)
			        vec[i] /= sum;
        }

        template<typename T>
        inline void NormTwoNormalize(vector<T>& vec)
        {
            double sum = 0;

            for (T item : vec)
                sum += item * item;

            if (sum != 0)
	        {
		        double root = sqrt(sum);
		        for (int i = vec.size() - 1; i >= 0; i--)
		        {
			        vec[i] /= root;
		        }
	        }
        }

		inline vector<int> RandomPickUp(int cardNum, int pickUpNum)
		{
			assert(cardNum >= pickUpNum);
			vector<int> result;

			if (cardNum != pickUpNum)
			{
				int* cards = new int[cardNum];
				for (int i = 0; i < cardNum; i++)
					cards[i] = i;

				for (int i = 0; i < pickUpNum; i++)
				{
					int index = (double)rand() * (cardNum - i - 1) / RAND_MAX + i;
					assert(index >= 0 && index < cardNum);
					swap(cards[i], cards[index]);
				}

				for (int i = 0; i < pickUpNum; i++)
					result.push_back(cards[i]);
				sort(result.begin(), result.end());

				delete[] cards;
			}
			else
			{
				for (int i = 0; i < cardNum; i++)
					result.push_back(i);
			}

			return result;
		}

		template<typename T>
		inline double GaussianDistance(const vector<T>& u, const vector<T>& v, double sigma)
		{
			assert(u.size() == v.size());

			double distance = 0;
			for (int i = u.size() - 1; i >= 0; i--)
				distance += (u[i] - v[i]) * (u[i] - v[i]);

			return exp(-distance / (2 * sigma * sigma));
		}
    }
}