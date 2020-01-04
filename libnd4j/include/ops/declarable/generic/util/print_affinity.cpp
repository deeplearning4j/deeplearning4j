/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
#if NOT_EXCLUDED(OP_print_affinity)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/print_variable.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(print_affinity, 1, 1, true, 0, 0) {
            // TODO: make this op compatible with ArrayList etc
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            nd4j_printf("<Node %i>: Actuality: [HOST: %s; DEVICE: %s]; affinity: [%i]; Pointers: [HOST: %p; DEVICE: %p]; DataBuffer length: %lld\n", block.nodeId(), input->isActualOnHostSide() ? "true" : "false", input->isActualOnDeviceSide() ? "true" : "false", input->dataBuffer()->deviceId(), input->getBuffer(), input->getSpecialBuffer(), input->dataBuffer()->getLenInBytes());

            return Status::OK();
        }

        DECLARE_TYPES(print_affinity) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, nd4j::DataType::ANY)
                    ->setAllowedInputTypes(1, {ALL_STRINGS})
                    ->setAllowedOutputTypes(0, nd4j::DataType::INT32);
        }

        DECLARE_SHAPE_FN(print_affinity) {
            return SHAPELIST(ConstantShapeHelper::getInstance()->scalarShapeInfo(DataType::INT32));
        }
    }
}

#endif