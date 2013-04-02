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

	public ref class EdgeMatching
	{
    public:
        static List<Tuple<List<Point>^, Mat<float>^>^>^ GetHitmap(Mat<uchar>^ image, bool thinning)
        {
            vector<pair<vector<NativePoint>, NativeMat>> result = 
                PerformHitmap(Convertor::ToNativeMat(image), thinning);

            List<Tuple<List<Point>^, Mat<float>^>^>^ list = 
                gcnew List<Tuple<List<Point>^, Mat<float>^>^>();
            for (int i = 0; i < result.size(); i++)
            {
                const vector<NativePoint>& item1 = result[i].first;
                const NativeMat& item2 = result[i].second;

                List<Point>^ points = gcnew List<Point>();
                for (int j = 0; j < item1.size(); j++)
                    points->Add(Point(item1[j]));

                Mat<float>^ mat = Convertor::ToManagedMat(item2);

                list->Add(Tuple::Create(points, mat));
            }

            return list;
        }
    };
}
