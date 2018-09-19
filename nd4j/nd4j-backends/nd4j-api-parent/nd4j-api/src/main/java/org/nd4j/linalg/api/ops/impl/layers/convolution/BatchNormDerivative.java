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

package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * BatchNormDerivative operation
 */
@Slf4j
public class BatchNormDerivative extends BatchNorm {

    @Builder(builderMethodName = "derivativeBuilder")
    public BatchNormDerivative(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays,
                               INDArray[] outputArrays, boolean inPlace, boolean applyGamma,
                               boolean applyBeta, double epsilon, int[] axis) {
        super(sameDiff, inputFunctions, inputArrays, outputArrays, inPlace, applyGamma, applyBeta, epsilon, axis);
    }

    public BatchNormDerivative() {}


    @Override
    public String opName() {
        return "batchnorm_bp";
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op name found for " + opName());
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name found for " + opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
