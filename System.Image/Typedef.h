#pragma once

#include "../System/Type.h"
using namespace std;

namespace System
{
    namespace Image
    {
        typedef Vector<double> Descriptor;
        typedef Vector<float> Descriptor_f;

        typedef Vector<Descriptor> LocalFeature;
        typedef Vector<Descriptor_f> LocalFeature_f;

        typedef Vector<double> GlobalFeature;
        typedef Vector<float> GlobalFeature_f;

        typedef Vector<double> Word;
        typedef Vector<float> Word_f;

        typedef Vector<double> Histogram;
        typedef Vector<float> Histogram_f;

        inline void Convert(const LocalFeature& src, LocalFeature_f& dst)
        {
            dst.clear();

            for (auto descriptor : src)
            {
                Descriptor_f tmp;
                for (auto item : descriptor)
                    tmp.push_back((float)item);

                dst.push_back(tmp);
            }
        }
    }
}