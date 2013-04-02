// ClrAdapter.h

#pragma once

#include "../Export/Export.h"
using namespace System;
using namespace System::Collections::Generic;

namespace ClrAdapter {

    public value struct Point
    {
    public:
        Point(int x, int y)
        {
            this->x = x;
            this->y = y;
        }

        Point(const NativePoint& point)
        {
            this->x = point.x;
            this->y = point.y;
        }

        int x, y;
    };

    generic<typename T>
    public ref class Mat
    {
    public:
        Mat(int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;
            this->data = gcnew array<T, 2>(rows, cols);
        }

        property int Rows
        {
            int get()
            {
                return rows;
            }
        }

        property int Cols
        {
            int get()
            {
                return cols;
            }
        }

        property T default[int, int]
        {
            T get(int r, int c)
            {
                return data[r, c];
            }

            void set(int r, int c, T value)
            {
                data[r, c] = value;
            }
        }

    private:
        array<T, 2>^ data;
        int rows, cols;
    };

    public ref class Convertor
    {
    public:
        static Mat<float>^ ToManagedMat(const NativeMat& mat)
        {
            Mat<float>^ result = gcnew Mat<float>(mat.rows, mat.cols);

            for (int i = 0; i < mat.rows; i++)
                for (int j = 0; j < mat.cols; j++)
                    result[i, j] = mat.atFLOAT(i, j);

            return result;
        }

        static NativeMat ToNativeMat(Mat<uchar>^ mat)
        {
            NativeMat result(mat->Rows, mat->Cols, EPT_UCHAR);

            for (int i = 0; i < result.rows; i++)
                for (int j = 0; j < result.cols; j++)
                    result.atUCHAR(i, j) = mat[i, j];

            return result;
        }
    };

    typedef List<Tuple<List<Point>^, Mat<float>^>^>^ Info;

	public ref class EdgeMatching
	{
    public:
        static array<Info>^ GetHitmap(List<Mat<uchar>^>^ images, bool thinning)
        {
            array<Info>^ result = gcnew array<Info>(images->Count);

            vector<NativeMat> nativeMats;
            for (int i = 0; i < images->Count; i++)
                nativeMats.push_back(Convertor::ToNativeMat(images[i]));

            vector<NativeInfo> tmp = PerformHitmap(nativeMats, thinning);

            for (int i = 0; i < images->Count; i++)
            {
                for (int j = 0; j < tmp[i].size(); j++)
                {
                    const vector<NativePoint>& item1 = tmp[i][j].first;
                    const NativeMat& item2 = tmp[i][j].second;

                    List<Point>^ points = gcnew List<Point>();
                    for (int k = 0; k < item1.size(); k++)
                        points->Add(Point(item1[k]));

                    Mat<float>^ mat = Convertor::ToManagedMat(item2);

                    result[i]->Add(Tuple::Create(points, mat));
                }
            }

            return result;
        }
    };
}
