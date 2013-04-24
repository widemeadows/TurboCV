#pragma once

#include "../System/Type.h"

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        typedef ArrayList<double> Descriptor;
        typedef ArrayList<float> Descriptor_f;

        typedef ArrayList<Descriptor> LocalFeatureVec;
        typedef ArrayList<Descriptor_f> LocalFeature_f;

        typedef ArrayList<double> GlobalFeatureVec;
        typedef ArrayList<float> GlobalFeature_f;

        typedef ArrayList<double> Word;
        typedef ArrayList<float> Word_f;

        typedef ArrayList<double> Histogram;
        typedef ArrayList<float> Histogram_f;

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
}