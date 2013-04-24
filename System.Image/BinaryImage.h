#pragma once

#include <cv.h>
#include <queue>
using namespace cv;
using namespace std;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        // Assume: Edgels are 1 and Background is 0.
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

        // Assume: Edgels are 1 and Background is 0.
        inline void clean(InputArray input, OutputArray output, int points = 1)
        {
            assert(input.type() == CV_8U);

            Mat src = input.getMat();
            src.copyTo(output);

            Mat dst = output.getMat();
            Mat cur = src.clone();

            static int dy[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
            static int dx[] = { -1, 0, 1, -1, 1, -1, 0, 1 };

            for (int i = 0; i < cur.rows; i++)
            {
                for (int j = 0; j < cur.cols; j++)
                {
                    if (cur.at<uchar>(i, j))
                    {
                        queue<Point> q;
                        vector<Point> component;

                        q.push(Point(j, i));
                        cur.at<uchar>(i, j) = 0;

                        while (!q.empty())
                        {
                            Point front = q.front();
                            q.pop();
                            component.push_back(front);

                            for (int k = 0; k < 8; k++)
                            {
                                int newY = front.y + dy[k], newX = front.x + dx[k];

                                if (newY >= 0 && newY < cur.rows && 
                                    newX >= 0 && newX < cur.cols &&
                                    cur.at<uchar>(newY, newX))
                                {
                                    q.push(Point(newX, newY));
                                    cur.at<uchar>(newY, newX) = 0;
                                }
                            }
                        }

                        if (component.size() <= points)
                        {
                            for (size_t k = 0; k < component.size(); k++)
                                dst.at<uchar>(component[k].y, component[k].x) = 0;
                        }
                    }
                }
            }
        }

        // Assume: Edgels are 1 and Background is 0.
        inline ArrayList<Point> GetEdgels(const Mat& sketchImage)
        {
	        ArrayList<Point> points;

	        for (int i = 0; i < sketchImage.rows; i++)
		        for (int j = 0; j < sketchImage.cols; j++)
			        if (sketchImage.at<uchar>(i, j))
				        points.push_back(Point(j, i));

	        return points;
        }
    }
}
}