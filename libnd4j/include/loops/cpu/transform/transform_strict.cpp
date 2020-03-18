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
//  @author  raver119@gmail.com
//

#include <system/op_boilerplate.h>
#include <helpers/Loops.h>
#include <types/types.h>
#include <loops/transform_strict.h>
#include <loops/legacy_ops.h>

using namespace simdOps;

namespace functions {
    namespace transform {

        template <typename X>
        void TransformStrict<X>::exec(
				int opNum,
				void *x,
				Nd4jLong *xShapeInfo,
				void *z,
				Nd4jLong *zShapeInfo,
				void *extraParams, uint64_t threadId, uint64_t numThreads) {
                    DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xShapeInfo, z, zShapeInfo, extraParams, threadId, numThreads), TRANSFORM_STRICT_OPS);
		}

        template <typename X>
        template<typename OpType>
		void _CUDA_H TransformStrict<X>::exec(
                    void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vz,
                    Nd4jLong *zShapeInfo,
                    void *vextraParams, uint64_t threadId, uint64_t numThreads) {

            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<X *>(vz);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            sd::TransformLoops<X,X,X>::template loopTransform<OpType>(x, xShapeInfo, z, zShapeInfo, extraParams, threadId, numThreads);
        }

        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT TransformStrict, , FLOAT_TYPES);
    }
}