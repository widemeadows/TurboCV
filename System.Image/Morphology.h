#pragma once

#include <cv.h>
using namespace cv;

namespace System
{
    namespace Image
    {
        inline void thin(InputArray input, OutputArray output, int iterations = 100) 
        {
            assert(input.type() == CV_8U);
            Mat src = input.getMat();
            src.copyTo(output);

            Mat dst = output.getMat();
	        for (int n = 0; n < iterations; n++) 
	        {
		        Mat prev = dst.clone();
		        for (int i = 0; i < src.rows; i++) 
		        {
			        for (int j = 0; j < src.cols; j++) 
			        {
				        if (prev.at<uchar>(i, j) == 1) 
				        {
					        int ap = 0;
					        int p2 = (i == 0) ? 0 : prev.at<uchar>(i - 1, j);
					        int p3 = (i == 0 || j == src.cols - 1) ? 0 : prev.at<uchar>(i - 1, j + 1);
					        if (p2 == 0 && p3 == 1) 
						        ap++;

					        int p4 = (j == src.cols - 1) ? 0 : prev.at<uchar>(i, j + 1);
					        if (p3 == 0 && p4 == 1) 
						        ap++;

					        int p5 = (i == src.rows - 1 || j == src.cols - 1) ? 0 : prev.at<uchar>(i + 1, j + 1);
					        if (p4 == 0 && p5 == 1) 
						        ap++;

					        int p6 = (i == src.rows - 1) ? 0 : prev.at<uchar>(i + 1, j);
					        if (p5 == 0 && p6 == 1) 
						        ap++;

					        int p7 = (i == src.rows - 1 || j == 0) ? 0 : prev.at<uchar>(i + 1, j - 1);
					        if (p6 == 0 && p7 == 1) 
						        ap++;

					        int p8 = (j == 0) ? 0 : prev.at<uchar>(i, j - 1);
					        if (p7 == 0 && p8 == 1) 
						        ap++;

					        int p9 = (i == 0 || j == 0) ? 0 : prev.at<uchar>(i - 1, j - 1);
					        if (p8 == 0 && p9 == 1) 
						        ap++;
					        if (p9 == 0 && p2 == 1) 
						        ap++;

					        if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7)
						        if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
							        dst.at<uchar>(i, j) = 0;
				        }
			        }
		        }

		        prev = dst.clone();
		        for (int i = 0; i < src.rows; i++) 
		        {
			        for (int j = 0; j < src.cols; j++)
			        {
				        if (prev.at<uchar>(i, j) == 1)
				        {
					        int ap = 0;
					        int p2 = (i == 0) ? 0 : prev.at<uchar>(i - 1, j);
					        int p3 = (i == 0 || j == src.cols - 1) ? 0 : prev.at<uchar>(i - 1, j + 1);
					        if (p2 == 0 && p3 == 1) 
						        ap++;

					        int p4 = (j == src.cols - 1) ? 0 : prev.at<uchar>(i, j + 1);
					        if (p3 == 0 && p4 == 1) 
						        ap++;

					        int p5 = (i == src.rows - 1 || j == src.cols - 1) ? 0 : prev.at<uchar>(i + 1, j + 1);
					        if (p4 == 0 && p5 == 1) 
						        ap++;

					        int p6 = (i == src.rows - 1) ? 0 : prev.at<uchar>(i + 1, j);
					        if (p5 == 0 && p6 == 1) 
						        ap++;

					        int p7 = (i == src.rows - 1 || j == 0) ? 0 : prev.at<uchar>(i + 1, j - 1);
					        if (p6 == 0 && p7 == 1) 
						        ap++;

					        int p8 = (j == 0) ? 0 : prev.at<uchar>(i, j - 1);
					        if (p7 == 0 && p8 == 1) 
						        ap++;

					        int p9 = (i == 0 || j == 0) ? 0 : prev.at<uchar>(i - 1, j - 1);
					        if (p8 == 0 && p9 == 1) 
						        ap++;
					        if (p9 == 0 && p2 == 1) 
						        ap++;

					        if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7) 
						        if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0) 
							        dst.at<uchar>(i, j) = 0;
				        }
			        }
		        }
	        }
        }
    }
}