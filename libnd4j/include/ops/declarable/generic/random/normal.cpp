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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_normal)

#include <ops/declarable/headers/random.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(random_normal, 1, 1, true, 2, 0) {
            // normal distribution
            auto rng = block.getRNG();
            // FIXME: to be implemented
/*
            REQUIRE_TRUE(rng != nullptr, 0, "RNG isn't defined for this Graph instance");

            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            functions::random::RandomFunction<T>::template execTransform<randomOps::GaussianDistribution<T>>(block.getRNG(), z->getBuffer(), z->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());
*/
            return Status::OK();
        }

        DECLARE_SHAPE_FN(random_normal) {
            auto in = INPUT_VARIABLE(0);
            auto shape = in->template asVectorT<Nd4jLong>();

            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), Nd4jLong);
            shape::shapeBuffer(shape.size(), shape.data(), newShape);

            return SHAPELIST(newShape);
        }
		
		DECLARE_SYN(randomnormal, random_normal);
    }
}

#endif