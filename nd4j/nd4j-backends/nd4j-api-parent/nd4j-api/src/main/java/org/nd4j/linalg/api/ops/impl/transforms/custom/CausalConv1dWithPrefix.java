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
 * Causal depthwise 1D convolution with per-prefix history checkpoints.
 *
 * Same convolution as {@link CausalConv1d} but requires the actualLen scalar and
 * additionally produces:
 *
 *   output 2: prefix [W, B, D, K-1] time-leading C-order, where slot t holds the
 *   retained history AFTER consuming input rows 0..t (the last K-1 elements of
 *   concat(stateIn, x[0:t+1]), using RAW convolution inputs).
 *
 * This is the capture primitive for accepted-prefix state selection in bundled MTP
 * speculative decoding: a partially accepted verification window commits
 * prefix[consumed-1] instead of re-executing the target for the consumed prefix.
 *
 * Inputs:
 *   0: x        [B, L, D]
 *   1: weight   [D, K] or [K, D]
 *   2+: bias [D], stateIn [B, D, K-1], actualLen INT64 scalar (REQUIRED)
 *
 * Integer arguments (mirroring causal_conv1d):
 *   0: activation (0=none, 1=silu)
 *   1: wFormat (0=[D,K], 1=[K,D])
 *
 * Outputs:
 *   0: output    [B, L, D]
 *   1: state_out [B, D, K-1]
 *   2: prefix    [W, B, D, K-1]  (W = L at shape time)
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class CausalConv1dWithPrefix extends DynamicCustomOp {

    public CausalConv1dWithPrefix(INDArray x, INDArray weight, INDArray bias,
                                  INDArray stateIn, INDArray actualLen) {
        this(x, weight, bias, stateIn, actualLen, 0, 0);
    }

    public CausalConv1dWithPrefix(INDArray x, INDArray weight, INDArray bias,
                                  INDArray stateIn, INDArray actualLen,
                                  int activation, int wFormat) {
        super(buildInputs(x, weight, bias, stateIn, actualLen), null);
        addIArgument(activation, wFormat);
    }

    public CausalConv1dWithPrefix(SameDiff sd, SDVariable x, SDVariable weight,
                                  SDVariable stateIn, SDVariable actualLen) {
        this(sd, x, weight, stateIn, actualLen, 0, 0);
    }

    public CausalConv1dWithPrefix(SameDiff sd, SDVariable x, SDVariable weight,
                                  SDVariable bias, SDVariable stateIn, SDVariable actualLen,
                                  int activation, int wFormat) {
        super(null, sd, buildSdInputs(x, weight, bias, stateIn, actualLen));
        addIArgument(activation, wFormat);
    }

    public CausalConv1dWithPrefix(SameDiff sd, SDVariable x, SDVariable weight,
                                  SDVariable stateIn, SDVariable actualLen,
                                  int activation, int wFormat) {
        super(null, sd, buildSdInputs(x, weight, stateIn, actualLen));
        addIArgument(activation, wFormat);
    }

    private static INDArray[] buildInputs(INDArray x, INDArray weight, INDArray bias,
                                          INDArray stateIn, INDArray actualLen) {
        List<INDArray> inputs = new ArrayList<>();
        inputs.add(x);
        inputs.add(weight);
        if (bias != null) inputs.add(bias);
        if (stateIn != null) inputs.add(stateIn);
        inputs.add(actualLen);
        return inputs.toArray(new INDArray[0]);
    }

    private static SDVariable[] buildSdInputs(SDVariable x, SDVariable weight, SDVariable bias,
                                              SDVariable stateIn, SDVariable actualLen) {
        List<SDVariable> inputs = new ArrayList<>();
        inputs.add(x);
        inputs.add(weight);
        if (bias != null) inputs.add(bias);
        if (stateIn != null) inputs.add(stateIn);
        inputs.add(actualLen);
        return inputs.toArray(new SDVariable[0]);
    }

    private static SDVariable[] buildSdInputs(SDVariable x, SDVariable weight,
                                              SDVariable stateIn, SDVariable actualLen) {
        return buildSdInputs(x, weight, null, stateIn, actualLen);
    }

    @Override
    public String opName() {
        return "causal_conv1d_with_prefix";
    }

    /**
     * Output dtype mirrors the native contract: output 0 uses the activation dtype;
     * state_out and prefix use the state input's dtype when present (the native
     * selector builds on stateType, which defaults to x's dtype only when stateIn
     * is absent).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType activationType = inputDataTypes.get(0);
        // Inputs: 0=x, 1=weight, [2=bias], [state], actualLen (always last).
        DataType stateType = activationType;
        int numInputs = inputDataTypes.size();
        // stateIn, when present, is the input immediately before the final actualLen.
        if (numInputs >= 4) {
            stateType = inputDataTypes.get(numInputs - 2);
        }
        return Arrays.asList(activationType, stateType, stateType);
    }

    @Override
    public int getNumOutputs() {
        return 3;
    }
}
