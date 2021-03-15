/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver on 6/12/2018.
//

#include <types/types.h>
#include <system/op_boilerplate.h>
#include <loops/type_conversions.h>
#include <helpers/OmpLaunchHelper.h>
#include <execution/Threads.h>

namespace sd {

    template <typename T>
    _CUDA_H void TypeCast::convertFromQuantized(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz) {
        //
        auto z = reinterpret_cast<T *>(dz);

        auto fx = reinterpret_cast<float *>(dx);
        auto amin = sd::math::nd4j_abs<float>(fx[0]);
        auto amax = sd::math::nd4j_abs<float>(fx[1]);


        auto x = reinterpret_cast<char *>(dx) + 8;


        for (Nd4jLong e = 0; e < N; e++) {
            z[e] = static_cast<T>(static_cast<float>(x[e]) / static_cast<float>(DataTypeUtils::max<int8_t>()) * sd::math::nd4j_max<float>(amin, amax));
        }
    }

    template <typename T>
    _CUDA_H void TypeCast::convertToQuantized(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz) {
        // find min/max first

        auto x = reinterpret_cast<T *>(dx);
        auto z = reinterpret_cast<char *>(dz);

        T mn = DataTypeUtils::max<T>();
        T mx = -DataTypeUtils::max<T>();

        for (Nd4jLong e = 0; e < N; e++) {
            T v = x[e];
            if (v < mn)
                mn = v;

            if (v > mx)
                mx = v;
        }

        // we shift by 2 fp32 elements
        auto rz = z + 8;

        //
        auto fz = reinterpret_cast<float *>(z);

        float max = static_cast<float>(mx);
        float min = static_cast<float>(mn);

        int max_byte = static_cast<int>(DataTypeUtils::max<int8_t>());
        fz[0] = min;
        fz[1] = max;

        auto amax = sd::math::nd4j_abs<float>(max);
        auto amin = sd::math::nd4j_abs<float>(min);

        // now we actually apply quantization
        auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e++) {
                rz[e] = static_cast<char>(sd::math::nd4j_round<float, char>( 1.0f * static_cast<float>(x[e]) / sd::math::nd4j_max<float>(amax, amin) * max_byte));
            }
        };

        samediff::Threads::parallel_for(func,  0, N);
    }

    template <typename T>
    void TypeCast::convertToThreshold(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz) {
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

#ifdef _OPENMP
        int threads = OmpLaunchHelper::betterThreads(N);
        auto span = OmpLaunchHelper::betterSpan(N, threads);
#else
        int threads = 1;
        auto span = N;
#endif


        T tt = static_cast<T>(threshold);
        T mtt = -tt;

        // we use 3 as offset, since first 12 bytes are occupied with header
        int flimit = limit + 4;
        volatile int cnt = 4;
        volatile bool flag = false;
        PRAGMA_OMP_PARALLEL_THREADS(threads)
        {
            int tid = omp_get_thread_num();
            int start = span * tid;
            int stop = span * (tid + 1);
            if (stop > l)
                stop = l;

            for (int e = start; e < stop; e++) {
                bool flag_load;
PRAGMA_OMP_ATOMIC_ARGS(read)
                flag_load = flag;
                if (flag_load)
                    break;

                T cUpd = x[e];
                if (cUpd >= tt) {
                    int idx;
PRAGMA_OMP_ATOMIC_ARGS(capture)
                    idx = cnt++;

                    if (idx >= flimit) {
PRAGMA_OMP_ATOMIC_ARGS(write)
                        flag = true;
                        break;
                    }

                    z[idx] = e + 1;
                    x[e] -= tt;
                } else if (cUpd <= mtt) {
                    int idx;
PRAGMA_OMP_ATOMIC_ARGS(capture)
                    idx = cnt++;

                    if (idx >= flimit) {
PRAGMA_OMP_ATOMIC_ARGS(write)
                        flag = true;
                        break;
                    }


                    z[idx] = -e - 1;
                    x[e] += tt;
                }
            }
        }
    }

    template <typename T>
    void TypeCast::convertFromThreshold(Nd4jPointer * extras, const void *dx, Nd4jLong N, void *dz) {
        FloatBits fb;
        auto z = reinterpret_cast<T *>(dz);
        auto x = reinterpret_cast<const int *>(dx);
        int limit = x[0];
        fb.i_ = x[2];
        float threshold = fb.f_;

        // we use 3 as offset, since first 12 bytes are occupied with header
        int flimit = limit + 4;

        auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e++) {
                int el = x[e];
                int ael = sd::math::nd4j_abs<int>(el) - 1;
                z[ael] += el > 0 ? static_cast<T>(threshold) : static_cast<T>(-threshold);
            }
        };

        samediff::Threads::parallel_for(func,  4, flimit);
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

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                z[i] = static_cast<T>(static_cast<float>(x[i]));
            }
        };
        samediff::Threads::parallel_for(func,  0, N);
    };

    template void TypeCast::convertFromThreshold<double>(Nd4jPointer * extras, const void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertFromThreshold<float>(Nd4jPointer * extras, const void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertFromThreshold<float16>(Nd4jPointer * extras, const void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertFromThreshold<bfloat16>(Nd4jPointer * extras, const void *dx, Nd4jLong N, void *dz);

    template void TypeCast::convertToThreshold<double>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertToThreshold<float>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertToThreshold<float16>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertToThreshold<bfloat16>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);

    template void TypeCast::convertFromQuantized<double>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertFromQuantized<float>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertFromQuantized<float16>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);

    template void TypeCast::convertToQuantized<double>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertToQuantized<float>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);
    template void TypeCast::convertToQuantized<float16>(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);

#ifndef __CLION_IDE__
    BUILD_DOUBLE_TEMPLATE(template void TypeCast::convertGeneric, (Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz), LIBND4J_TYPES, LIBND4J_TYPES)
#endif
}
