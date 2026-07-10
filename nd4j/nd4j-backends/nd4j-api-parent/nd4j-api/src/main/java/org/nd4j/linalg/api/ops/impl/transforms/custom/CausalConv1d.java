/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Depthwise causal 1D convolution with state management.
 *
 * Used in Gated Delta Networks (GDN) and Mamba architectures.
 * Performs a causal (left-padded) depthwise 1D convolution.
 *
 * Inputs:
 *   0: x         [B, L, D]
 *   1: weight    [D, K] (wFormat=0) or [K, D] (wFormat=1)
 *   2: bias      [D] (optional)
 *   3: stateIn   [B, D, K-1] (optional)
 *   4: actualLen scalar INT64, real sequence length for stateOut (optional)
 *
 * Int args:
 *   0: activation (0=none, 1=SiLU/Swish)
 *   1: wFormat   (0=[D,K] PyTorch/ONNX default, 1=[K,D] TensorFlow)
 *
 * Outputs:
 *   0: output    [B, L, D]
 *   1: stateOut  [B, D, K-1]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class CausalConv1d extends DynamicCustomOp {

    public CausalConv1d(INDArray x, INDArray weight, INDArray bias, INDArray stateIn, int activation) {
        this(x, weight, bias, stateIn, null, activation, 0);
    }

    public CausalConv1d(INDArray x, INDArray weight, INDArray bias, INDArray stateIn, int activation, int wFormat) {
        this(x, weight, bias, stateIn, null, activation, wFormat);
    }

    public CausalConv1d(INDArray x, INDArray weight, INDArray bias, INDArray stateIn,
                        INDArray actualLen, int activation, int wFormat) {
        super(buildInputs(x, weight, bias, stateIn, actualLen), null);
        addIArgument(activation, wFormat);
    }

    public CausalConv1d(SameDiff sd, SDVariable x, SDVariable weight, SDVariable bias,
                        SDVariable stateIn, int activation) {
        this(sd, x, weight, bias, stateIn, null, activation, 0);
    }

    public CausalConv1d(SameDiff sd, SDVariable x, SDVariable weight, SDVariable bias,
                        SDVariable stateIn, int activation, int wFormat) {
        this(sd, x, weight, bias, stateIn, null, activation, wFormat);
    }

    public CausalConv1d(SameDiff sd, SDVariable x, SDVariable weight, SDVariable bias,
                        SDVariable stateIn, SDVariable actualLen, int activation, int wFormat) {
        super(null, sd, buildSdInputs(x, weight, bias, stateIn, actualLen));
        addIArgument(activation, wFormat);
    }

    private static INDArray[] buildInputs(INDArray x, INDArray weight, INDArray bias, INDArray stateIn,
                                          INDArray actualLen) {
        List<INDArray> inputs = new ArrayList<>();
        inputs.add(x);
        inputs.add(weight);
        if (bias != null) inputs.add(bias);
        if (stateIn != null) inputs.add(stateIn);
        if (actualLen != null) inputs.add(actualLen);
        return inputs.toArray(new INDArray[0]);
    }

    private static SDVariable[] buildSdInputs(SDVariable x, SDVariable weight, SDVariable bias, SDVariable stateIn,
                                              SDVariable actualLen) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(x);
        inputs.add(weight);
        if (bias != null) inputs.add(bias);
        if (stateIn != null) inputs.add(stateIn);
        if (actualLen != null) inputs.add(actualLen);
        return inputs.toArray(new SDVariable[0]);
    }

    @Override
    public String opName() {
        return "causal_conv1d";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(0);
        return Arrays.asList(dt, dt);
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
