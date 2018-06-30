//
// Created by raver on 6/30/2018.
//

#include <helpers/OmpLaunchHelper.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace nd4j {
    Nd4jLong OmpLaunchHelper::betterSpan(Nd4jLong N) {
        return OmpLaunchHelper::betterSpan(N, OmpLaunchHelper::betterThreads(N));
    }

    Nd4jLong OmpLaunchHelper::betterSpan(Nd4jLong N, Nd4jLong numThreads) {
        auto t = N / numThreads;

        return 0L;
    }

    int OmpLaunchHelper::betterThreads(Nd4jLong N) {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
}
