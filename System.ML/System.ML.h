#pragma once

#include "KNN.h"
#include "MQDF.h"
#include "TSNE.h"
#include "SNE.h"
#include "../System/System.h"
using namespace TurboCV::System;

ArrayList<ArrayList<double>> PerformTSNE(
    const ArrayList<ArrayList<double>>& samples,
    int dimension = 2);

ArrayList<ArrayList<double>> PerformSNE(
    const ArrayList<ArrayList<double>>& samples,
    int dimension = 2);