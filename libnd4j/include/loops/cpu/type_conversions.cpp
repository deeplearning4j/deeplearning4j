//
// Created by raver on 6/12/2018.
//

#include <types/types.h>
#include <op_boilerplate.h>
#include <loops/type_conversions.h>

namespace nd4j {

    template<typename T>
    void TypeCast::convertToThreshold(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz) {
        // we suppose that first 4 bytes are integer, second 4 bytes are float
        // integer: enc length
        // integer: dec length
        // float: threshold
        FloatBits fb;
        auto x = reinterpret_cast<T *>(dx);
        auto z = reinterpret_cast<int *>(dz);
        int limit = z[0];
        fb.i_ = z[2];
        float threshold = fb.f_;

        // TODO: int limit is sad thing here, 2B elements limitation
        auto l = static_cast<int>(N);
        z[1] = l;

        // we use 3 as offset, since first 12 bytes are occupied with header
        int flimit = limit + 4;
        volatile int cnt = 4;
        volatile bool flag = false;
#pragma omp parallel for schedule(guided) default(shared)
        for (int e = 0; e < l; e++) {
            bool flag_load;
#pragma omp atomic read
            flag_load = flag;
            if (flag_load)
                continue;

            T cUpd = x[e];
            if (cUpd >= static_cast<T>(threshold)) {
                int idx;
#pragma omp atomic capture
                idx = cnt++;

                if (idx >= flimit) {
#pragma omp atomic write
                    flag = true;
                    continue;
                }

                z[idx] = e + 1;
                x[e] -= static_cast<T>(threshold);
            } else if (cUpd <= static_cast<T>(-threshold)) {
                int idx;
#pragma omp atomic capture
                idx = cnt++;

                if (idx >= flimit) {
#pragma omp atomic write
                    flag = true;
                    continue;
                }

                z[idx] = -e - 1;
                x[e] += static_cast<T>(threshold);
            }
        }
    }

    template<typename T>
    void TypeCast::convertFromThreshold(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz) {
        FloatBits fb;
        auto z = reinterpret_cast<T *>(dz);
        auto x = reinterpret_cast<int *>(dx);
        int limit = x[0];
        fb.i_ = x[2];
        float threshold = fb.f_;

        // we use 3 as offset, since first 12 bytes are occupied with header
        int flimit = limit + 4;

#pragma omp parallel for schedule(guided)
        for (int e = 4; e < flimit; e++) {
            int el = x[e];
            int ael = nd4j::math::nd4j_abs<int>(el) - 1;
            z[ael] += el > 0 ? threshold : -threshold;
        }
    }


    /**
     * This is cpu version, so leave it here as inline, to avoid templates instantiation
     *
     * @tparam S
     * @tparam T
     * @param dx
     * @param N
     * @param dz
     */
    template<typename S, typename T>
    void TypeCast::convertGeneric(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz) {
        auto x = reinterpret_cast<S *>(dx);
        auto z = reinterpret_cast<T *>(dz);

        if (N < nd4j::Environment::getInstance()->elementwiseThreshold()) {
#pragma omp simd
            for (int i = 0; i < N; i++) {
                z[i] = static_cast<T>(x[i]);
            }
        } else {

#pragma omp parallel for
            for (int i = 0; i < N; i++) {
                z[i] = static_cast<T>(x[i]);
            }
        }
    };


    template void TypeCast::convertFromThreshold<float>(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz);

    template void TypeCast::convertFromThreshold<float16>(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz);

    template void TypeCast::convertFromThreshold<double>(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz);

    template void TypeCast::convertToThreshold<float>(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz);

    template void TypeCast::convertToThreshold<float16>(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz);

    template void TypeCast::convertToThreshold<double>(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz);

    BUILD_DOUBLE_TEMPLATE(template void TypeCast::convertGeneric, (Nd4jPointer * extras, void * dx, Nd4jLong N, void * dz), LIBND4J_TYPES, LIBND4J_TYPES)
}
