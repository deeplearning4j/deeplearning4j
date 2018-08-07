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
            NDArray<T>* counts = INPUT_VARIABLE(0);
            NDArray<T>* means = INPUT_VARIABLE(1);
            NDArray<T>* variances = INPUT_VARIABLE(2);

            NDArray<T>* resMeans = OUTPUT_VARIABLE(0);
            NDArray<T>* resVariances = OUTPUT_VARIABLE(1);
/*
    divisor = math_ops.reciprocal(counts, name="divisor")
    if shift is not None:
      shifted_mean = math_ops.multiply(mean_ss, divisor, name="shifted_mean")
      mean = math_ops.add(shifted_mean, shift, name="mean")
    else:  # no shift.
      shifted_mean = math_ops.multiply(mean_ss, divisor, name="mean")
      mean = shifted_mean
    variance = math_ops.subtract(
        math_ops.multiply(variance_ss, divisor),
        math_ops.square(shifted_mean),
        name="variance")
  return (mean, variance)

*/

            T shift(0);
            
            if (block.getTArguments()->size() > 0) {
                shift = T_ARG(0);
            }
//            nd4j_printf("means output %p \n",  resMeans->getBuffer())
//            nd4j_printf("variance output %p \n",  resVariances->getBuffer())
//            resMeans->printBuffer("Output Means");
//            resVariances->printBuffer("Output Variance");

            means->template applyScalar<simdOps::Divide<T>>((*counts)(0), resMeans, nullptr);

            std::unique_ptr<NDArray<T>> squareMeans(resMeans->dup('c'));
            std::unique_ptr<NDArray<T>> tempVariances(resVariances->dup('c'));

            squareMeans->template applyTransform<simdOps::Square<T>>((T*)nullptr);
            variances->template applyScalar<simdOps::Divide<T>>((*counts)(0), tempVariances.get(), nullptr);
//            tempVariances->printIndexedBuffer("varianced divided by count");
            tempVariances->template applyPairwiseTransform<simdOps::Subtract<T>>(squareMeans.get(), resVariances, nullptr);

            if (shift != T(0)) {
                resMeans->template applyScalar<simdOps::Add<T>>(shift, resMeans, nullptr);
            }
          
//            resMeans->printBuffer("Output Means");
//            resVariances->printBuffer("Output Variance");
//            nd4j_printf("means output %p \n",  resMeans->getBuffer())
//            nd4j_printf("variance output %p \n",  resVariances->getBuffer())
            return ND4J_STATUS_OK;
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