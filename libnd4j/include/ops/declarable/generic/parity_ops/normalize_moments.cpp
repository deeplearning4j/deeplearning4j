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
// Created by george@skymind.io on 26.01.2018.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_normalize_moments)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(normalize_moments, 3, 2, false, 1, 0) {
            auto counts = INPUT_VARIABLE(0);
            auto  means = INPUT_VARIABLE(1);
            auto variances = INPUT_VARIABLE(2);

            auto resMeans = OUTPUT_VARIABLE(0);
            auto resVariances = OUTPUT_VARIABLE(1);

            // FIXME: double?
            double shift(0);
            
            if (block.getTArguments()->size() > 0) {
                shift = T_ARG(0);
            }

            means->applyScalar(scalar::Divide, *counts, resMeans, nullptr);

            std::unique_ptr<NDArray> squareMeans(resMeans->dup('c'));
            std::unique_ptr<NDArray> tempVariances(resVariances->dup('c'));

            squareMeans->applyTransform(transform::Square, squareMeans.get(), nullptr);
            variances->applyScalar(scalar::Divide, *counts, tempVariances.get(), nullptr);
//            tempVariances->printIndexedBuffer("varianced divided by count");
            tempVariances->applyPairwiseTransform(pairwise::Subtract, squareMeans.get(), resVariances, nullptr);

            if (shift != 0) {
                resMeans->applyScalar(scalar::Add, shift, resMeans, nullptr);
            }

            return Status::OK();
        }

        DECLARE_SHAPE_FN(normalize_moments) {
            auto in = inputShape->at(1);

            Nd4jLong* meanShape = nullptr;
            Nd4jLong* varianceShape = nullptr;

            COPY_SHAPE_EX(in, meanShape, block.getWorkspace());
            COPY_SHAPE_EX(in, varianceShape, block.getWorkspace());

            auto shapeList = SHAPELIST(); 
            shapeList->push_back(meanShape);
            shapeList->push_back(varianceShape);

            return shapeList;
        }
    }

}

#endif