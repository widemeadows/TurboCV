#pragma once

#include <cv.h>
using namespace cv;

namespace System
{
    namespace Image
    {
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
    }
}