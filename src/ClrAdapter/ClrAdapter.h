// ClrAdapter.h

#pragma once

#include "../System/System.h"
#include "../Export/Export.h"
#using <System.Runtime.Serialization.dll>

using namespace TurboCV::System;
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
        template<typename T>
        static Mat<T>^ ToManagedMat(const NativeMat& mat, T type)
        {
            Mat<T>^ result = gcnew Mat<T>(mat.rows, mat.cols);

            for (int i = 0; i < mat.rows; i++)
                for (int j = 0; j < mat.cols; j++)
                    result[i, j] = mat.at<T>(i, j);

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

        static NativeMat ToNativeMat(Mat<float>^ mat)
        {
            NativeMat result(mat->Rows, mat->Cols, EPT_FLOAT);

            for (int i = 0; i < result.rows; i++)
                for (int j = 0; j < result.cols; j++)
                    result.at<float>(i, j) = mat[i, j];

            return result;
        }
    };

	public ref class EdgeMatching
	{
    public:
        enum class Type
        {
            OCM,
            HIT
        };

        typedef List<Tuple<List<Point>^, Object^>^> Info;

        static array<Info^>^ Train(Type type, List<Mat<uchar>^>^ images, bool thinning)
        {
            ArrayList<NativeMat> nativeMats;
            for (int i = 0; i < images->Count; i++)
                nativeMats.Add(Convertor::ToNativeMat(images[i]));

            ArrayList<NativeInfo> tmp;
            if (type == Type::OCM)
                tmp = EdgeMatchingPredict(EPT_OCM, nativeMats, thinning);
            else
                tmp = EdgeMatchingPredict(EPT_HIT, nativeMats, thinning);

            array<Info^>^ result = gcnew array<Info^>(images->Count);
            for (int i = 0; i < images->Count; i++)
            {
                result[i] = gcnew Info();

                for (int j = 0; j < tmp[i].Count(); j++)
                {
                    const ArrayList<NativePoint>& item1 = tmp[i][j].Item1();
                    const NativeMat& item2 = tmp[i][j].Item2();

                    List<Point>^ points = gcnew List<Point>();
                    for (int k = 0; k < item1.Count(); k++)
                        points->Add(Point(item1[k]));

                    if (type == Type::OCM)
                    {
                        Mat<float>^ mat = Convertor::ToManagedMat(item2, float());
                        result[i]->Add(Tuple::Create(points, (Object^)mat));
                    }
                    else
                    {
                        Mat<uchar>^ mat = Convertor::ToManagedMat(item2, uchar());
                        result[i]->Add(Tuple::Create(points, (Object^)mat));
                    }
                }
            }

            return result;
        }

        static Info^ GetMap(Type type, Mat<uchar>^ image, bool thinning)
        {
            NativeInfo tmp;
            if (type == Type::OCM)
                tmp = EdgeMatchingPredict(EPT_OCM, Convertor::ToNativeMat(image), thinning);
            else
                tmp = EdgeMatchingPredict(EPT_HIT, Convertor::ToNativeMat(image), thinning);

            Info^ result = gcnew Info();
            for (int i = 0; i < tmp.Count(); i++)
            {
                const ArrayList<NativePoint>& item1 = tmp[i].Item1();
                const NativeMat& item2 = tmp[i].Item2();

                List<Point>^ points = gcnew List<Point>();
                for (int j = 0; j < item1.Count(); j++)
                    points->Add(Point(item1[j]));

                if (type == Type::OCM)
                {
                    Mat<float>^ mat = Convertor::ToManagedMat(item2, float());
                    result->Add(Tuple::Create(points, (Object^)mat));
                }
                else
                {
                    Mat<uchar>^ mat = Convertor::ToManagedMat(item2, uchar());
                    result->Add(Tuple::Create(points, (Object^)mat));
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
            RHOG,
            SC,
            RSC,
            PSC,
            RPSC,
            HOOSC,
            RHOOSC
        };

        typedef List<float> Word;
        typedef List<double> Histogram; 

        static Tuple<List<Word^>^, List<Histogram^>^>^ Train(
            Type type, List<Mat<uchar>^>^ images, int wordNum, bool thinning)
        {
            LocalFeatureType nativeType;

            switch (type)
            {
            case Type::RHOG:
                nativeType = EPT_RHOG;
                break;
            case Type::SC:
                nativeType = EPT_SC;
                break;
            case Type::RSC:
                nativeType = EPT_RSC;
                break;
            case Type::PSC:
                nativeType = EPT_PSC;
                break;
            case Type::RPSC:
                nativeType = EPT_RPSC;
                break;
            case Type::HOOSC:
                nativeType = EPT_HOOSC;
                break;
            case Type::RHOOSC:
                nativeType = EPT_RHOOSC;
                break;
            default:
                break;
            }

            ArrayList<NativeMat> nativeMats;
            for (int i = 0; i < images->Count; i++)
                nativeMats.Add(Convertor::ToNativeMat(images[i]));

            Group<ArrayList<NativeWord>, ArrayList<NativeHistogram>> result = 
                LocalFeatureTrain(nativeType, nativeMats, wordNum, thinning);
            ArrayList<NativeWord>& nativeWords = result.Item1();
            ArrayList<NativeHistogram>& nativeHistograms = result.Item2();

            List<Word^>^ words = gcnew List<Word^>();
            for (int i = 0; i < nativeWords.Count(); i++)
            {
                Word^ word = gcnew Word();
                for (int j = 0; j < nativeWords[i].Count(); j++)
                    word->Add(nativeWords[i][j]);

                words->Add(word);
            }

            List<Histogram^>^ histograms = gcnew List<Histogram^>();
            for (int i = 0; i < nativeHistograms.Count(); i++)
            {
                Histogram^ histogram = gcnew Histogram();
                for (int j = 0; j < nativeHistograms[i].Count(); j++)
                    histogram->Add(nativeHistograms[i][j]);

                histograms->Add(histogram);
            }

            return Tuple::Create(words, histograms);
        }

        static Histogram^ GetLocalFeature(
            Type type, Mat<uchar>^ image, List<Word^>^ words, bool thinning)
        {
            LocalFeatureType nativeType;

            switch (type)
            {
            case Type::RHOG:
                nativeType = EPT_RHOG;
                break;
            case Type::SC:
                nativeType = EPT_SC;
                break;
            case Type::RSC:
                nativeType = EPT_RSC;
                break;
            case Type::PSC:
                nativeType = EPT_PSC;
                break;
            case Type::RPSC:
                nativeType = EPT_RPSC;
                break;
            case Type::HOOSC:
                nativeType = EPT_HOOSC;
                break;
            case Type::RHOOSC:
                nativeType = EPT_RHOOSC;
                break;
            default:
                break;
            }

            ArrayList<NativeWord> nativeWords(words->Count);
            for (int i = 0; i < words->Count; i++)
                for (int j = 0; j < words[i]->Count; j++)
                    nativeWords[i].Add(words[i][j]);

            NativeHistogram nativeHistogram = LocalFeaturePredict(nativeType, 
                Convertor::ToNativeMat(image), nativeWords, thinning);

            Histogram^ histogram = gcnew Histogram();
            for (int i = 0; i < nativeHistogram.Count(); i++)
                histogram->Add(nativeHistogram[i]);

            return histogram;
        }
    };

	public ref class GlobalFeature
	{
	public:
		enum class Type
		{
			GHOG,
			GIST
		};

		typedef List<double> Vec; 

		static List<Vec^>^ Train(Type type, List<Mat<uchar>^>^ images, bool thinning)
		{
			GlobalFeatureType nativeType;

			switch (type)
			{
			case Type::GHOG:
				nativeType = EPT_GHOG;
				break;
			case Type::GIST:
				nativeType = EPT_GIST;
				break;
			default:
				break;
			}

			ArrayList<NativeMat> nativeMats;
			for (int i = 0; i < images->Count; i++)
				nativeMats.Add(Convertor::ToNativeMat(images[i]));

			ArrayList<NativeVec> nativeVecs = 
				GlobalFeaturePredict(nativeType, nativeMats, thinning);

			List<Vec^>^ vecs = gcnew List<Vec^>();
			for (int i = 0; i < nativeVecs.Count(); i++)
			{
				Vec^ vec = gcnew Vec();
				for (int j = 0; j < nativeVecs[i].Count(); j++)
					vec->Add(nativeVecs[i][j]);

				vecs->Add(vec);
			}

			return vecs;
		}

		static Vec^ GetGlobalFeature(Type type, Mat<uchar>^ image, bool thinning)
		{
			GlobalFeatureType nativeType;

			switch (type)
			{
			case Type::GHOG:
				nativeType = EPT_GHOG;
				break;
			case Type::GIST:
				nativeType = EPT_GIST;
				break;
			default:
				break;
			}

			NativeVec nativeVec = 
				GlobalFeaturePredict(nativeType, Convertor::ToNativeMat(image), thinning);

			Vec^ vec = gcnew Vec();
			for (int i = 0; i < nativeVec.Count(); i++)
				vec->Add(nativeVec[i]);

			return vec;
		}
	};
}
