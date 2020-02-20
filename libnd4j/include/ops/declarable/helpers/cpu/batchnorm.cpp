/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include<ops/declarable/helpers/batchnorm.h>
#include <helpers/ShapeUtils.h>
#include <OmpLaunchHelper.h>
#include <execution/Threads.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void batchnorm_(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta,
                       NDArray* output,
                       const std::vector<int>& axes, const double epsilon) {

    // formula: output = gamma * ((input - mean) / sqrt(variance + epsilon)) + beta

    const T* x = input->bufferAsT<T>();
          T* z = output->bufferAsT<T>();
    const T* m = mean->bufferAsT<T>();
    const T* v = variance->bufferAsT<T>();
    const T* g = gamma == nullptr ? nullptr : gamma->bufferAsT<T>();
    const T* b = beta  == nullptr ? nullptr : beta->bufferAsT<T>();

    const bool xzSameOffset = shape::haveSameShapeAndStrides(input->getShapeInfo(), output->getShapeInfo());

    bool paramSameOffset = shape::haveSameShapeAndStrides(mean->getShapeInfo(), variance->getShapeInfo());
    if(paramSameOffset && gamma != nullptr)
        paramSameOffset &= shape::haveSameShapeAndStrides(mean->getShapeInfo(), gamma->getShapeInfo());
    if(paramSameOffset && beta != nullptr)
        paramSameOffset &= shape::haveSameShapeAndStrides(mean->getShapeInfo(), beta->getShapeInfo());

    const Nd4jLong  lenBig        = input->lengthOf();
    const Nd4jLong  lenSmall      = mean->lengthOf();

    const Nd4jLong steps = lenBig / lenSmall;
    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(input->rankOf(), axes);

    OmpLaunchHelper info(lenBig, lenSmall);

    auto func = PRAGMA_THREADS_DO {

        Nd4jLong* xOffsets = new Nd4jLong[steps];
        Nd4jLong* zOffsets = xzSameOffset ? xOffsets : new Nd4jLong[steps];
        Nd4jLong* auxBuff = new Nd4jLong[2 * input->rankOf()];

        for (int j = 0; j < lenSmall; ++j) {

            const bool isOwner = (j < info._numThreads) ? thread_id == j : thread_id == (j % info._numThreads);

            if(!isOwner)
                continue;

            const auto meanOffset = shape::getIndexOffset(j, mean->getShapeInfo());
            const auto varOffset  = paramSameOffset ? meanOffset : shape::getIndexOffset(j, variance->getShapeInfo());

            const auto meanVal = m[meanOffset];
            auto sigmaInvGam   = static_cast<T>(1) / nd4j::math::nd4j_sqrt<T, T>(v[varOffset] + epsilon);

            if(g != nullptr) {
                const auto gammaOffset = paramSameOffset ? meanOffset : shape::getIndexOffset(j, gamma->getShapeInfo());
                sigmaInvGam *= g[gammaOffset];
            }

            T betaVal = static_cast<T>(0);
            if(b != nullptr) {
                const auto betaOffset = paramSameOffset ? meanOffset : shape::getIndexOffset(j, beta->getShapeInfo());
                betaVal = b[betaOffset];
            }

            // calculate offsets for input and output
            shape::outerArrayOffsets(xOffsets, j, input->getShapeInfo(), mean->getShapeInfo(), auxBuff, dimsToExclude.data());
            if(!xzSameOffset)
                shape::outerArrayOffsets(zOffsets, j, output->getShapeInfo(), mean->getShapeInfo(), auxBuff, dimsToExclude.data());

            PRAGMA_OMP_SIMD
            for (uint i = 0; i < steps; ++i)
                z[zOffsets[i]] = (x[xOffsets[i]] - meanVal) * sigmaInvGam + betaVal;
        }

        delete []auxBuff;
        delete []xOffsets;
        if(!xzSameOffset)
            delete []zOffsets;
    };

    samediff::Threads::parallel_do(func, info._numThreads);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void batchnorm2_(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta,
                        NDArray* output,
                        const std::vector<int>& axes, const double epsilon) {

    // formula: output = gamma * ((input - mean) / sqrt(variance + epsilon)) + beta

    const auto x = input->bufferAsT<T>();
          auto z = output->bufferAsT<T>();
    const auto m = mean->bufferAsT<T>();
    const auto v = variance->bufferAsT<T>();
    const auto g = gamma == nullptr ? nullptr : gamma->bufferAsT<T>();
    const auto b = beta  == nullptr ? nullptr : beta->bufferAsT<T>();

    // xRank == zRank, minRank = meanRank = varianceRank = gammaRank = betaRank
    const uint xRank   = input->rankOf();
    const uint minRank = mean->rankOf();
    const uint numAxes = axes.size();

    const bool xzSameOffset = shape::haveSameShapeAndStrides(input->getShapeInfo(), output->getShapeInfo());

    bool paramSameOffset = shape::haveSameShapeAndStrides(mean->getShapeInfo(), variance->getShapeInfo());
    if(paramSameOffset && gamma != nullptr)
        paramSameOffset &= shape::haveSameShapeAndStrides(mean->getShapeInfo(), gamma->getShapeInfo());
    if(paramSameOffset && beta != nullptr)
        paramSameOffset &= shape::haveSameShapeAndStrides(mean->getShapeInfo(), beta->getShapeInfo());

    auto func = PRAGMA_THREADS_FOR {

        Nd4jLong coords[MAX_RANK];

        for (auto i = start; i < stop; i++) {

            shape::index2coords(i, input->getShapeInfo(), coords);

            const auto xOffset = shape::getOffset(input->getShapeInfo(), coords);
            const auto zOffset = xzSameOffset ? xOffset : shape::getOffset(output->getShapeInfo(), coords);

            if(minRank == xRank) {
                for (uint i = 0, j = 0; i < xRank; ++i) {
                    if(j < numAxes && i != axes[j])
                        coords[i] = 0;
                    else
                        ++j;
                }
            }
            else    // minRank = numAxes = 1 in this case
                coords[0] = coords[axes[0]];

            const auto meanOffset     = shape::getOffset(mean->getShapeInfo(), coords);
            const auto varianceOffset = paramSameOffset ? meanOffset : shape::getOffset(variance->getShapeInfo(), coords);

            T sigmaInvGam = 1. / nd4j::math::nd4j_sqrt<T, T>(v[varianceOffset] + epsilon);

            if(g != nullptr) {
                const auto gammaOffset = paramSameOffset ? meanOffset : shape::getOffset(gamma->getShapeInfo(), coords);
                sigmaInvGam *= g[gammaOffset];
            }

            z[zOffset] = (x[xOffset] - m[meanOffset]) * sigmaInvGam;

            if(b != nullptr) {
                const auto betaOffset = paramSameOffset ? meanOffset : shape::getOffset(beta->getShapeInfo(), coords);
                z[zOffset] += b[betaOffset];
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, input->lengthOf());
}

//////////////////////////////////////////////////////////////////////////
void batchnorm(const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& axes, const double epsilon) {

    // batchnorm2_ is slower
    BUILD_SINGLE_SELECTOR(input->dataType(), batchnorm_, (input, mean, variance, gamma, beta, output, axes, epsilon), FLOAT_TYPES);
}



BUILD_SINGLE_TEMPLATE(template void batchnorm_, (const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* beta, NDArray* output, const std::vector<int>& axes, const double epsilon), FLOAT_TYPES);

}
}
}

