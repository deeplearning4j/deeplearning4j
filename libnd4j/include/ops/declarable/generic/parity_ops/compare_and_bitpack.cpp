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
 // @author sgazeos@gmail.com
 //

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/headers/datatypes.h>
#include <ops/declarable/helpers/transforms.h>
#include <array/NDArrayFactory.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(compare_and_bitpack, 2, 1, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            sd::ops::helpers::compareAndBitpack(block, *x, *y, *z);
            return Status::OK();
        }

        DECLARE_TYPES(compare_and_bitpack) {
            getOpDescriptor()
                ->setAllowedInputTypes(0, DataType::ANY)
                ->setAllowedInputTypes(1, DataType::ANY)
                ->setAllowedOutputTypes(0, DataType::UINT8);
        }

        DECLARE_SHAPE_FN(compare_and_bitpack) {
            auto inShape = inputShape->at(0);
            auto shapes = shape::shapeOf(inShape);
            const int rank = shape::rank(inShape);
            REQUIRE_TRUE(!shape::isScalar(inShape), 0, "Input should not be a scalar");
            std::vector<Nd4jLong> shapeDims {shapes, shapes + rank};
            REQUIRE_TRUE(shapeDims[rank-1] % 8 ==0 , 0, "Last dimension of the input (which is %i) should be divisible by 8 ", shapeDims[rank-1]);
            shapeDims[rank-1] = shapeDims[rank-1] / 8 ;
            DataType newType = DataType::UINT8;
            auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(newType, shape::order(inShape), shapeDims);
            return SHAPELIST(outputShape);
        }

    }
}