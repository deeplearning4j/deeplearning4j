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
//  @author  raver119@gmail.com
//

#include <system/op_boilerplate.h>
#include <helpers/Loops.h>
#include <types/types.h>
#include <loops/transform_bool.h>
#include <loops/legacy_ops.h>

using namespace simdOps;

namespace functions {
    namespace transform {

        template <typename X, typename Y>
        void TransformBool<X, Y>::exec(int opNum,
                                       const void *x, const Nd4jLong *xShapeInfo,
                                       void *z, const Nd4jLong *zShapeInfo,
                                       void *extraParams,
                                       uint64_t threadId, uint64_t numThreads) {
                    DISPATCH_BY_OPNUM_TT(exec, PARAMS(x, xShapeInfo, z, zShapeInfo, extraParams, threadId, numThreads), TRANSFORM_BOOL_OPS);
		}

        template <typename X, typename Z>
        template<typename OpType>
		void _CUDA_H TransformBool<X, Z>::exec(const void *vx, const Nd4jLong *xShapeInfo,
		                                       void *vz, const Nd4jLong *zShapeInfo,
		                                       void *vextraParams,
		                                       uint64_t threadId, uint64_t numThreads) {

            auto x = reinterpret_cast<const X *>(vx);
		    auto z = reinterpret_cast<Z *>(vz);
		    auto extraParams = reinterpret_cast<X *>(vextraParams);

            sd::TransformLoops<X,Z,X>::template loopTransform<OpType>(x, xShapeInfo, z, zShapeInfo, extraParams, threadId, numThreads);
        }

        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT TransformBool, , LIBND4J_TYPES, BOOL_TYPES);
    }
}