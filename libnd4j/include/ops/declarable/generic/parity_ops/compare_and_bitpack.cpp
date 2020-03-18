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
// @author sgazeos@gmail.com
//

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/headers/datatypes.h>
#include <array/NDArrayFactory.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(compare_and_bitpack, 2, 1, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);
            auto z0 = NDArrayFactory::create<bool>(x->ordering(), x->getShapeAsVector());
            BROADCAST_CHECK_EMPTY(x, y, (&z0));
            
            auto tZ = BroadcastHelper::broadcastApply(BROADCAST_BOOL(GreaterThan), x, y, &z0);
            bitcast res;
            auto status = res.execute({tZ}, {z}, {}, {DataType::UINT8}, {}, {}, false);
            if (tZ != &z0) {
                delete tZ;
            }
            
            return status;
        }

        DECLARE_TYPES(compare_and_bitpack) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, DataType::ANY)
                    ->setAllowedInputTypes(1, DataType::ANY)
                    ->setAllowedOutputTypes(0, DataType::UINT8);
        }

        DECLARE_SHAPE_FN(compare_and_bitpack) {
            auto inShape = inputShape->at(0);
            DataType newType = DataType::UINT8;

            return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(ShapeDescriptor(inShape, newType)));
        }

    }
}