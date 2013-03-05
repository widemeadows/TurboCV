#pragma once

#include "../System/System.h"
#include <cv.h>
#include <highgui.h>
#include <tuple>
using namespace cv;
using namespace std;

namespace System
{
    namespace Image
    {
        typedef vector<double> Histogram;

        const int INF = 2147483647;
        const double EPS = 1e-14;
        const int MAX_GRAYSCALE = 255;

        bool operator<(const Point& u, const Point& v)
        {
            if (u.x < v.x)
                return true;
            else if (u.x == v.x)
                return u.y < v.y;
            else
                return false;
        }

        template<typename T1, typename T2>
        bool operator<(const tuple<T1, T2>& u, const tuple<T1, T2>& v)
        {
            if (get<0>(u) < get<0>(v))
                return true;
            else if (get<0>(u) == get<0>(v))
                return get<1>(u) < get<1>(v);
            else
                return false;
        }

        struct PointHash
        {
            size_t operator()(const Point& v) const
            {
                return v.y * 10000000 + v.x;
            }
        };

        inline Mat reverse(Mat grayScaleImage) 
        {
            assert(grayScaleImage.type() == CV_8U);
            Mat result(grayScaleImage.size(), CV_8U);

            for (int i = 0; i < grayScaleImage.rows; i++)
                for (int j = 0; j < grayScaleImage.cols; j++)
                    result.at<uchar>(i, j) = MAX_GRAYSCALE - grayScaleImage.at<uchar>(i, j);

            return result;
        }

        template<typename RandomAccessIterator>
        inline void NormOneNormalize(const RandomAccessIterator& begin, const RandomAccessIterator& end)
        {
            double sum = 0;
            RandomAccessIterator curr = begin;

            do
            {
                sum += *curr;     
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
        inline vector<T> RandomPickUp(const vector<T>& vec, int pickUpNum)
        {
            int cardNum = vec.size();
            assert(cardNum >= pickUpNum);

            vector<T> pickUps;
            vector<int> randomIndexes = RandomPickUp(cardNum, pickUpNum);
            int counter = 0;

            for (int i = 0; i < cardNum; i++)
            {
                if (counter < randomIndexes.size() && randomIndexes[counter] == i)
                {
                    counter++;
                    pickUps.push_back(vec[i]);
                }
            }

            return pickUps;
        }

        template<typename T>
        inline tuple<vector<T>, vector<T>> RandomSplit(const vector<T>& vec, int pickUpNum)
        {
            int cardNum = vec.size();
            assert(cardNum >= pickUpNum);

            vector<T> pickUps, others;
            vector<int> randomIndexes = RandomPickUp(cardNum, pickUpNum);
            int counter = 0;

            for (int i = 0; i < cardNum; i++)
            {
                if (counter < randomIndexes.size() && randomIndexes[counter] == i)
                {
                    counter++;
                    pickUps.push_back(vec[i]);
                }
                else
                    others.push_back(vec[i]);
            }

            return make_tuple<pickUps, others>;
        }

        inline void imshow(const Mat& image, bool scale = true)
        {
            double maximum = 1e-14, minimum = 1e14;
            int type = image.type();
            Mat tmp(image.size(), CV_64F);

            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    double value;

                    if (type == CV_8U)
                        value = image.at<uchar>(i, j);
                    else if (type == CV_64F)
                        value = image.at<double>(i, j);
                    else if (type == CV_32S)
                        value = image.at<int>(i, j);
                    else if (type == CV_8S)
                        value = image.at<char>(i, j);
                    else if (type == CV_32F)
                        value = image.at<float>(i, j);

                    maximum = max(value, maximum);
                    minimum = min(value, minimum);
                    tmp.at<double>(i, j) = value;
                }
            }

            if (maximum > minimum)
            {
                for (int i = 0; i < tmp.rows; i++)
                    for (int j = 0; j < tmp.cols; j++)
                        tmp.at<double>(i, j) = (tmp.at<double>(i, j) - minimum) / 
                            (maximum - minimum);
            }

            imshow("OpenCV", tmp);
        }
    }
}