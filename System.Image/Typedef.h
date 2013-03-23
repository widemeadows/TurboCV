#pragma once

#include "../System/Type.h"
using namespace std;

namespace System
{
    namespace Image
    {
        typedef Vector<double> Descriptor;
        typedef Vector<float> Descriptor_f;

        typedef Vector<Descriptor> LocalFeatureVec;
        typedef Vector<Descriptor_f> LocalFeature_f;

        typedef Vector<double> GlobalFeatureVec;
        typedef Vector<float> GlobalFeature_f;

        typedef Vector<double> Word;
        typedef Vector<float> Word_f;

        typedef Vector<double> Histogram;
        typedef Vector<float> Histogram_f;

        inline void Convert(const LocalFeatureVec& src, LocalFeature_f& dst)
        {
            dst.clear();

            for (auto descriptor : src)
            {
                Descriptor_f tmp(descriptor.size());
                for (int i = 0; i < descriptor.size(); i++)
                    tmp[i] = (float)descriptor[i];

                dst.push_back(tmp);
            }
        }

        inline void Convert(const GlobalFeatureVec& src, GlobalFeature_f& dst)
        {
            dst.clear();

            for (auto item : src)
                dst.push_back((float)item);
        }
    }
}