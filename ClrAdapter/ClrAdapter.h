// ClrAdapter.h

#pragma once

#include "../Export/Export.h"
#using <System.Runtime.Serialization.dll>

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Runtime::Serialization;

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
    [DataContract]
    public ref class Mat
    {
    public:
        Mat(int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;
            this->data = gcnew array<T>(rows * cols);
        }

        [DataMember]
        property int Rows
        {
        public:
            int get()
            {
                return rows;
            }

        protected:
            void set(int value)
            {
                rows = value;
            }
        }
        
        [DataMember]
        property int Cols
        {
        public:
            int get()
            {
                return cols;
            }

        private:
            void set(int value)
            {
                cols = value;
            }
        }

        [DataMember]
        property array<T>^ Data
        {
        public:
            array<T>^ get()
            {
                return data;
            }

        private:
            void set(array<T>^ value)
            {
                data = value;
            }
        }

        property T default[int, int]
        {
            T get(int r, int c)
            {
                return data[r * cols + c];
            }

            void set(int r, int c, T value)
            {
                data[r * cols + c] = value;
            }
        }

    private:
        array<T>^ data;
        int rows, cols;
    };

    private ref class Convertor
    {
    public:
        static Mat<uchar>^ ToManagedMat(const NativeMat& mat)
        {
            Mat<uchar>^ result = gcnew Mat<uchar>(mat.rows, mat.cols);

            for (int i = 0; i < mat.rows; i++)
                for (int j = 0; j < mat.cols; j++)
                    result[i, j] = mat.at<uchar>(i, j);

            return result;
        }

        static NativeMat ToNativeMat(Mat<uchar>^ mat)
        {
            NativeMat result(mat->Rows, mat->Cols, EPT_UCHAR);

            for (int i = 0; i < result.rows; i++)
                for (int j = 0; j < result.cols; j++)
                    result.at<uchar>(i, j) = mat[i, j];

            return result;
        }
    };

	public ref class EdgeMatching
	{
    public:
        typedef List<Tuple<List<Point>^, Mat<uchar>^>^> Info;

        static Info^ GetHitmap(Mat<uchar>^ image, bool thinning)
        {
            NativeInfo tmp = PerformHitmap(Convertor::ToNativeMat(image), thinning);

            Info^ result = gcnew Info();

            for (int i = 0; i < tmp.size(); i++)
            {
                const vector<NativePoint>& item1 = tmp[i].first;
                const NativeMat& item2 = tmp[i].second;

                List<Point>^ points = gcnew List<Point>();
                for (int j = 0; j < item1.size(); j++)
                    points->Add(Point(item1[j]));

                Mat<uchar>^ mat = Convertor::ToManagedMat(item2);

                result->Add(Tuple::Create(points, mat));
            }

            return result;
        }

        static array<Info^>^ GetHitmap(List<Mat<uchar>^>^ images, bool thinning)
        {
            array<Info^>^ result = gcnew array<Info^>(images->Count);

            vector<NativeMat> nativeMats;
            for (int i = 0; i < images->Count; i++)
                nativeMats.push_back(Convertor::ToNativeMat(images[i]));

            vector<NativeInfo> tmp = PerformHitmap(nativeMats, thinning);

            for (int i = 0; i < images->Count; i++)
            {
                result[i] = gcnew List<Tuple<List<Point>^, Mat<uchar>^>^>();

                for (int j = 0; j < tmp[i].size(); j++)
                {
                    const vector<NativePoint>& item1 = tmp[i][j].first;
                    const NativeMat& item2 = tmp[i][j].second;

                    List<Point>^ points = gcnew List<Point>();
                    for (int k = 0; k < item1.size(); k++)
                        points->Add(Point(item1[k]));

                    Mat<uchar>^ mat = Convertor::ToManagedMat(item2);

                    result[i]->Add(Tuple::Create(points, mat));
                }
            }

            return result;
        }
    };

    public ref class LocalFeature
    {
    public:
        enum class Type
        {
            RHOG
        };

        typedef List<float> Word;
        typedef List<double> Histogram; 

        static Tuple<List<Word^>^, List<Histogram^>^>^ Train(Type type, List<Mat<uchar>^>^ images, int wordNum, bool thinning)
        {
            LocalFeatureType nativeType;

            switch (type)
            {
            case Type::RHOG:
                nativeType = EPT_RHOG;
                break;
            default:
                break;
            }

            vector<NativeMat> nativeMats;
            for (int i = 0; i < images->Count; i++)
                nativeMats.push_back(Convertor::ToNativeMat(images[i]));

            pair<vector<NativeWord>, vector<NativeHistogram>> result = LocalFeatureTrain(nativeType, nativeMats, wordNum, thinning);
            vector<NativeWord>& nativeWords = result.first;
            vector<NativeHistogram>& nativeHistograms = result.second;

            List<Word^>^ words = gcnew List<Word^>();
            for (int i = 0; i < nativeWords.size(); i++)
            {
                words[i] = gcnew Word();
                for (int j = 0; j < nativeWords[i].size(); j++)
                    words[i]->Add(nativeWords[i][j]);
            }

            List<Histogram^>^ histograms = gcnew List<Histogram^>();
            for (int i = 0; i < nativeHistograms.size(); i++)
            {
                histograms[i] = gcnew Histogram();
                for (int j = 0; j < nativeHistograms[i].size(); j++)
                    histograms[i]->Add(nativeHistograms[i][j]);
            }

            return Tuple::Create(words, histograms);
        }

        static Histogram^ GetLocalFeature(Type type, Mat<uchar>^ image, List<Word^>^ words, bool thinning)
        {
            LocalFeatureType nativeType;

            switch (type)
            {
            case Type::RHOG:
                nativeType = EPT_RHOG;
                break;
            default:
                break;
            }

            vector<NativeWord> nativeWords(words->Count);
            for (int i = 0; i < words->Count; i++)
                for (int j = 0; j < words[i]->Count; i++)
                    nativeWords[i].push_back(words[i][j]);

            NativeHistogram nativeHistogram = LocalFeaturePredict(nativeType, Convertor::ToNativeMat(image), 
                nativeWords, thinning);
            Histogram^ histogram = gcnew Histogram();
            for (int i = 0; i < nativeHistogram.size(); i++)
                histogram->Add(nativeHistogram[i]);

            return histogram;
        }
    };
}
