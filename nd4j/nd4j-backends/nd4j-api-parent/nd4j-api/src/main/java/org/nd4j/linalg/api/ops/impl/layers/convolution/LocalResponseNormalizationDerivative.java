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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;

import java.util.List;


/**
 * LocalResponseNormalizationDerivative operation
 */
@Slf4j
public class LocalResponseNormalizationDerivative extends LocalResponseNormalization {
    @Builder(builderMethodName = "derivativeBuilder")
    public LocalResponseNormalizationDerivative(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputs, INDArray[] outputs, boolean inPlace, LocalResponseNormalizationConfig config) {
        super(sameDiff, inputFunctions, inputs, outputs, inPlace, config);
    }

    public LocalResponseNormalizationDerivative() {}

    @Override
    public String opName() {
        return "lrn_bp";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
