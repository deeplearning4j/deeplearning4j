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
            NDArray shift = NDArrayFactory::create<double>(0.);

            if (block.getTArguments()->size() > 0) {
                shift.assign(T_ARG(0));
            }

            means->applyScalarArr(scalar::Divide, counts, resMeans, nullptr);

            NDArray* squareMeans = resMeans->dup('c');
            NDArray* tempVariances = resVariances->dup('c');

            squareMeans->applyTransform(transform::Square, squareMeans, nullptr);
            variances->applyScalarArr(scalar::Divide, counts, tempVariances, nullptr);
//            tempVariances->printIndexedBuffer("varianced divided by count");
            tempVariances->applyPairwiseTransform(pairwise::Subtract, squareMeans, resVariances, nullptr);

            if (shift.e<double>(0) != 0) {
                resMeans->applyScalarArr(scalar::Add, &shift, resMeans, nullptr);
            }

            delete squareMeans;
            delete tempVariances;

            return Status::OK();
        }

        DECLARE_SHAPE_FN(normalize_moments) {
            auto in = inputShape->at(1);

            Nd4jLong* meanShape = nullptr;
            Nd4jLong* varianceShape = nullptr;

            COPY_SHAPE_EX(in, meanShape, block.getWorkspace());
            COPY_SHAPE_EX(in, varianceShape, block.getWorkspace());

            auto shapeList = SHAPELIST();
            shapeList->push_back(CONSTANT(meanShape));
            shapeList->push_back(CONSTANT(varianceShape));

            return shapeList;
        }

        DECLARE_TYPES(normalize_moments) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }

}

#endif