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
// @author GS (sgazeos@gmail.com), created on 10/1/2018
//


#include<ops/declarable/helpers/cross.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void weightedCrossEntropyWithLogitsFunctor_(NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output) {

        T posWeight = weights->e<T>(0);

        auto mainRoutineT1 = LAMBDA_TT(_x, _z, posWeight) {
            T targetWeight = (1. + (posWeight - (T)1.f) * _z);
            return (1. - _z) * _x +
                   targetWeight * (nd4j::math::nd4j_log<T,T>((T)1.f + nd4j::math::nd4j_exp<T,T>(-nd4j::math::nd4j_abs(_x))) +
                                   nd4j::math::nd4j_max(-_x, T(0.f))
                   );
        };

        auto mainRoutineT2 = LAMBDA_TTT(_x, _z, _w) {
            return (((T)1.0 - _z) * _x) +
                   _w * (nd4j::math::nd4j_log<T,T>(T(1.) + nd4j::math::nd4j_exp<T,T>(-nd4j::math::nd4j_abs(_x))) +
                         nd4j::math::nd4j_max(-_x, T(0.f)));
        };


        if (weights->isScalar()) {
            const_cast<NDArray*>(input)->applyPairwiseLambda<T>(const_cast<NDArray*>(targets), mainRoutineT1, output);
        }
        else
        {
            std::unique_ptr<NDArray> targetVector(new NDArray(*weights));
            targetVector->applyScalar(scalar::Add, -1.f);

            std::unique_ptr<NDArray> targetTensor(new NDArray(*targets));
            *targetTensor = (*targetVector * *targetTensor) + T(1.f);
            const_cast<NDArray*>(input)->applyTriplewiseLambda<T>(const_cast<NDArray*>(targets), targetTensor.get(), mainRoutineT2, output);
        }
}

void weightedCrossEntropyWithLogitsFunctor(NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output) {
    BUILD_SINGLE_SELECTOR(targets->dataType(), weightedCrossEntropyWithLogitsFunctor_, (targets, input, weights, output), FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void weightedCrossEntropyWithLogitsFunctor_, (NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output), FLOAT_TYPES);

}
}
}