/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019-2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//

#include <ops/declarable/helpers/transforms.h>
#include <helpers/Loops.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template<typename X, typename Z>
static void mergeMaxIndex_(const std::vector<const NDArray*>& inArrs, NDArray& output) {

    const Nd4jLong numArgs = inArrs.size();
    auto x = inArrs[0];

    auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++) {
            X max = -DataTypeUtils::max<X>();
            Z idx = static_cast<Z>(0);

            for (Nd4jLong i = 0; i < numArgs; i++) {
                X v = inArrs[i]->t<X>(e);
                if (v > max) {
                    max = v;
                    idx = static_cast<Z>(i);
                }
            }
            // FIXME, use .r<Z>(e)
            output.t<Z>(e) = static_cast<Z>(idx);
        }
    };

    samediff::Threads::parallel_for(func, 0, x->lengthOf());
}

void mergeMaxIndex(sd::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
    BUILD_DOUBLE_SELECTOR(inArrs[0]->dataType(), output.dataType(), mergeMaxIndex_, (inArrs, output), LIBND4J_TYPES, INDEXING_TYPES);
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeMax_(const std::vector<const NDArray*>& inArrs, NDArray& output) {

    const Nd4jLong numArgs = inArrs.size();
    auto x = inArrs[0];

    auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++) {
            T max = -DataTypeUtils::max<T>();
            for (Nd4jLong i = 0; i < numArgs; i++) {
                T v = inArrs[i]->e<T>(e);
                if (v > max)
                    max = v;
            }
            output.p(e, max);
        }
    };

    samediff::Threads::parallel_for(func, 0, x->lengthOf());
}

void mergeMax(sd::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
    BUILD_SINGLE_SELECTOR(output.dataType(), mergeMax_, (inArrs, output), LIBND4J_TYPES);
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeMaxBp_(const std::vector<const NDArray*>& inArrs, std::vector<NDArray*>& outArrs) {

    // outArrs.size() == inArrs.size() - 1
    const Nd4jLong numArgs = outArrs.size();
    // last array is gradient
    const auto gradient = inArrs[numArgs]->bufferAsT<T>();
    auto length = inArrs[numArgs]->lengthOf();

    bool bSameOrderAndEws1 = (1 == inArrs[numArgs]->ews());

    if (bSameOrderAndEws1) {
        auto gradOrdering = inArrs[numArgs]->ordering();

        for (int i = 0; i < numArgs; ++i) {
            bSameOrderAndEws1 &= (gradOrdering == inArrs[i]->ordering());
            bSameOrderAndEws1 &= (1 == inArrs[i]->ews());
            bSameOrderAndEws1 &= (gradOrdering == outArrs[i]->ordering());
            bSameOrderAndEws1 &= (1 == outArrs[i]->ews());
        }
    }


    if(bSameOrderAndEws1){
        auto func = PRAGMA_THREADS_FOR{
            for (auto e = start; e < stop; e++) {
                 T max = -DataTypeUtils::max<T>();
                 Nd4jLong nMaxIndex = 0;
                 for (Nd4jLong i = 0; i < numArgs; i++) {
                     const T* v = inArrs[i]->bufferAsT<T>();
                     if (v[e] > max) {
                         max = v[e];
                         nMaxIndex = i;
                     }
                 }
                 T* z = outArrs[nMaxIndex]->bufferAsT<T>();
                 z[e] = gradient[e];
            }
        };

        samediff::Threads::parallel_for(func, 0, length);
        return;
    }

    auto gradShape = inArrs[numArgs]->shapeInfo();
    std::vector<bool> vbSameShaepeAndStrides(numArgs);
    for (int i = 0; i < numArgs; ++i) {
        vbSameShaepeAndStrides[i] = shape::haveSameShapeAndStrides(gradShape, inArrs[i]->shapeInfo());
    }

    auto func = PRAGMA_THREADS_FOR{

        int coords[MAX_RANK];
            for (auto e = start; e < stop; e++) {

                 shape::index2coordsCPU(start, e, gradShape, coords);

                 const auto gradOffset =  shape::getOffset(gradShape, coords);

                 T max = -DataTypeUtils::max<T>();
                 Nd4jLong nMaxIndex = 0;

                 for (Nd4jLong i = 0; i < numArgs; i++) {

                     const auto xOffset = vbSameShaepeAndStrides[i] ? gradOffset : shape::getOffset(inArrs[i]->shapeInfo(), coords);
                     const T* v = inArrs[i]->bufferAsT<T>();
                     if (v[xOffset] > max) {
                         max = v[xOffset];
                         nMaxIndex = i;
                     }
                 }

                const auto zOffset = vbSameShaepeAndStrides[nMaxIndex] ? gradOffset : shape::getOffset(outArrs[nMaxIndex]->shapeInfo(), coords);

                T* z = outArrs[nMaxIndex]->bufferAsT<T>();
                z[zOffset] = gradient[gradOffset];
            }
    };

    samediff::Threads::parallel_for(func, 0, length);
    return;
}

void mergeMaxBp(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, std::vector<NDArray*>& outArrs) {
    BUILD_SINGLE_SELECTOR(outArrs[0]->dataType(), mergeMaxBp_, (inArrs, outArrs), LIBND4J_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeAvg_(const std::vector<const NDArray*>& inArrs, NDArray& output) {
    const Nd4jLong numArgs = inArrs.size();
    const T factor = 1.f / numArgs;
    auto x = inArrs[0];

    auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++) {
            T sum = 0.;
            for (Nd4jLong i = 0; i < numArgs; i++) {
                T v = inArrs[i]->e<T>(e);
                sum += v;
            }
            output.p<T>(e, sum * factor);
        }
    };

    samediff::Threads::parallel_for(func, 0, x->lengthOf());
}

void mergeAvg(sd::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
    BUILD_SINGLE_SELECTOR(output.dataType(), mergeAvg_, (inArrs, output), LIBND4J_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeAvgBp_(const NDArray& gradient, std::vector<NDArray*>& outArrs) {

    const Nd4jLong numArgs = outArrs.size();

    auto func = PRAGMA_THREADS_FOR{
        for (auto e = start; e < stop; e++) {

            T v = gradient.e<T>(e) / numArgs;

            for (Nd4jLong i = 0; i < numArgs; i++) {
                outArrs[i]->p<T>(e, v);
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf());
}

void mergeAvgBp(sd::LaunchContext* context, const NDArray& gradient, std::vector<NDArray*>& outArrs) {
    BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAvgBp_, (gradient, outArrs), LIBND4J_TYPES);
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeAdd_(const std::vector<const NDArray*>& inArrs, NDArray& output) {

    const Nd4jLong numArgs = inArrs.size();
    auto x = inArrs[0];

    auto func = PRAGMA_THREADS_FOR {
        for (auto e = start; e < stop; e++) {
            T sum = (T) 0.f;
            for (Nd4jLong i = 0; i < numArgs; i++)
                sum += inArrs[i]->e<T>(e);

            output.p(e, sum);
        }
    };

    samediff::Threads::parallel_for(func, 0, x->lengthOf());
}
    void mergeAdd(sd::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), mergeAdd_, (inArrs, output), LIBND4J_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
template<typename T>
static void mergeAddBp_(const NDArray& gradient, std::vector<NDArray*>& outArrs) {

    const Nd4jLong numArgs = outArrs.size();

    auto func = PRAGMA_THREADS_FOR{
        for (auto e = start; e < stop; e++) {

            T v = gradient.e<T>(e);

            for (Nd4jLong i = 0; i < numArgs; i++) {
                outArrs[i]->p<T>(e, v);
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, gradient.lengthOf());
}

void mergeAddBp(sd::LaunchContext* context, const NDArray& gradient, std::vector<NDArray*>& outArrs) {
    BUILD_SINGLE_SELECTOR(gradient.dataType(), mergeAddBp_, (gradient, outArrs), LIBND4J_TYPES);
}


}
}
}
