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

        int x, y;
    };

    template<typename T>
    public ref class Mat
    {

    };

	public ref class EdgeMatching
	{
        List<Tuple<List<Point>^, Mat<float>^>^ PerformHitmap(Mat<uchar>^ image, bool thinning)
        {

        }
    };
}
