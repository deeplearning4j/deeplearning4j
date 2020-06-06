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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 17.05.2018
// @author raver119@gmail.com
//

#include <ops/declarable/helpers/percentile.h>
#include <array/NDArrayFactory.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <array/ResultSet.h>

namespace sd    {
namespace ops     {
namespace helpers {

    template <typename X>
    static _CUDA_G void percentileKernel(void *vx, const Nd4jLong *xTadShapeInfo, const Nd4jLong *xTadOffsets,
                                         const Nd4jLong numTads, const Nd4jLong tadLength,
                                         void *vz, const Nd4jLong *zShapeInfo, const Nd4jLong zLength,
                                         const Nd4jLong position) {
        for (int t = blockIdx.x; t < numTads; t += gridDim.x) {
            auto x = reinterpret_cast<X*>(vx) + xTadOffsets[t];
            auto z = reinterpret_cast<X*>(vz);


            // sort tad
            if (tadLength > 1) {
                for (int m = 0; m < tadLength; m++) {
                    if (m % 2 == 0) {
                        for (int tid = threadIdx.x; tid < tadLength; tid += blockDim.x) {
                            auto top = 2 * tid + 1;
                            if (top < tadLength) {
                                auto t0 = shape::getIndexOffset(top - 1, xTadShapeInfo);
                                auto t1 = shape::getIndexOffset(top, xTadShapeInfo);

                                if (x[t0] > x[t1]) {
                                    //swap values
                                    X dz0 = x[t0];
                                    x[t0] = x[t1];
                                    x[t1] = dz0;
                                }
                            }
                        }
                    } else {
                        for (int tid = threadIdx.x; tid < tadLength; tid += blockDim.x) {
                            auto top = 2 * tid + 2;
                            if (top < tadLength) {
                                auto t0 = shape::getIndexOffset(top - 1, xTadShapeInfo);
                                auto t1 = shape::getIndexOffset(top, xTadShapeInfo);

                                if (x[t0] > x[t1]) {
                                    //swap values
                                    X dz0 = x[t0];
                                    x[t0] = x[t1];
                                    x[t1] = dz0;
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            // saving final value
            if (threadIdx.x == 0)
                z[shape::getIndexOffset(t, zShapeInfo)] = x[shape::getIndexOffset(position, xTadShapeInfo)];
            __syncthreads();
        }
    }



    template <typename T>
    static void _percentile(sd::LaunchContext * context, const NDArray& input, NDArray& output, std::vector<int>& axis, const float q, const int interpolation) {
        const int inputRank = input.rankOf();

        if(axis.empty())
            for(int i=0; i<inputRank; ++i)
                axis.push_back(i);
        else
            shape::checkDimensions(inputRank, axis);

        auto tempArray = input.dup();
        auto packX = ConstantTadHelper::getInstance().tadForDimensions(tempArray.shapeInfo(), axis);

        auto tadLength = shape::length(packX.primaryShapeInfo());

        const float fraction = 1.f - q / 100.;
        Nd4jLong position = 0;

        switch(interpolation) {
            case 0: // lower
                position = static_cast<Nd4jLong>(math::nd4j_ceil<float,T>((tadLength - 1) * fraction));
                break;
            case 1: // higher
                position = static_cast<Nd4jLong>(math::nd4j_floor<float,T>((tadLength - 1) * fraction));
                break;
            case 2: // nearest
                position = static_cast<Nd4jLong>(math::nd4j_round<float,T>((tadLength - 1) * fraction));
                break;
        }
        position = tadLength - position - 1;

        percentileKernel<T><<<256, 512, 1024, *context->getCudaStream()>>>(tempArray.specialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), packX.numberOfTads(), tadLength, output.specialBuffer(), output.specialShapeInfo(), output.lengthOf(), position);

        sd::DebugHelper::checkErrorCode(context->getCudaStream(), "percentile");
    }

    void percentile(sd::LaunchContext * context, const NDArray& input, NDArray& output, std::vector<int>& axises, const float q, const int interpolation) {
        NDArray::prepareSpecialUse({&output}, {&input});

        BUILD_SINGLE_SELECTOR(input.dataType(), _percentile, (context, input, output, axises, q, interpolation), LIBND4J_TYPES);

        NDArray::registerSpecialUse({&output}, {&input});
    }

    BUILD_SINGLE_TEMPLATE(template void _percentile, (sd::LaunchContext * context, const NDArray& input, NDArray& output, std::vector<int>& axises, const float q, const int interpolation), LIBND4J_TYPES);

}
}
}