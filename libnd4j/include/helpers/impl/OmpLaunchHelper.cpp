//
// Created by raver on 6/30/2018.
//

#include <helpers/OmpLaunchHelper.h>
#include <Environment.h>
#include <templatemath.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace nd4j {
    Nd4jLong OmpLaunchHelper::betterSpan(Nd4jLong N) {
        return OmpLaunchHelper::betterSpan(N, OmpLaunchHelper::betterThreads(N));
    }

    Nd4jLong OmpLaunchHelper::betterSpan(Nd4jLong N, Nd4jLong numThreads) {
        auto r = N % numThreads;
        auto t = N / numThreads;
        
        if (r == 0)
            return t;
        else {
            // fuck alignment
            return t + 1;
        }
    }

    int OmpLaunchHelper::betterThreads(Nd4jLong N) {
        #ifdef _OPENMP
            return betterThreads(N, omp_get_max_threads());
        #else
            return 1;
        #endif
    }

    int OmpLaunchHelper::betterThreads(Nd4jLong N, int maxThreads) {
        auto t = Environment::getInstance()->elementwiseThreshold();
        if (N < t)
            return 1;
        else {
            return static_cast<int>(nd4j::math::nd4j_min<Nd4jLong>(N / t, maxThreads));
        }
    }
}
