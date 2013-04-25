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
            dst.Clear();

            for (auto descriptor : src)
            {
                Descriptor_f tmp(descriptor.Count());
                for (int i = 0; i < descriptor.Count(); i++)
                    tmp[i] = (float)descriptor[i];

                dst.Add(tmp);
            }
        }

        inline void Convert(const GlobalFeatureVec& src, GlobalFeature_f& dst)
        {
            dst.Clear();

            for (auto item : src)
                dst.Add((float)item);
        }
    }
}
}