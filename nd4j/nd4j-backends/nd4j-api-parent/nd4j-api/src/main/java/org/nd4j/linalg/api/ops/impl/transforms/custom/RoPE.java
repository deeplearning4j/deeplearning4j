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
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Rotary Positional Embedding (RoPE).
 *
 * Applies rotary position embeddings to the input tensor, as described in
 * "RoFormer: Enhanced Transformer with Rotary Position Embedding".
 *
 * Inputs:
 *   0: input tensor
 *
 * Integer args:
 *   0: mode (default 0)
 *   1: nPast
 *   2: nDims
 *
 * Float args:
 *   0: freqBase (default 10000.0)
 *   1: freqScale (default 1.0)
 *
 * Output:
 *   0: tensor with rotary embeddings applied
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class RoPE extends DynamicCustomOp {

    private int mode = 0;
    private int nPast = 0;
    private int nDims = 0;
    private double freqBase = 10000.0;
    private double freqScale = 1.0;

    public RoPE(SameDiff sameDiff, SDVariable input, int mode, int nPast, int nDims,
                double freqBase, double freqScale) {
        super(null, sameDiff, new SDVariable[]{input}, false);
        this.mode = mode;
        this.nPast = nPast;
        this.nDims = nDims;
        this.freqBase = freqBase;
        this.freqScale = freqScale;
        addIArgument(mode, nPast, nDims);
        addTArgument(freqBase, freqScale);
    }

    public RoPE(INDArray input, int mode, int nPast, int nDims,
                double freqBase, double freqScale) {
        super(new INDArray[]{input}, null);
        this.mode = mode;
        this.nPast = nPast;
        this.nDims = nDims;
        this.freqBase = freqBase;
        this.freqScale = freqScale;
        addIArgument(mode, nPast, nDims);
        addTArgument(freqBase, freqScale);
    }

    public RoPE(INDArray input) {
        super(new INDArray[]{input}, null);
        addIArgument(mode, nPast, nDims);
        addTArgument(freqBase, freqScale);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) {
            this.mode = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.nPast = iArguments.get(1).intValue();
        }
        if (iArguments.size() > 2) {
            this.nDims = iArguments.get(2).intValue();
        }
        if (!tArguments.isEmpty()) {
            this.freqBase = tArguments.get(0);
        }
        if (tArguments.size() > 1) {
            this.freqScale = tArguments.get(1);
        }
    }

    @Override
    public String opName() {
        return "rope";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        Preconditions.checkState(first.isFPType(),
                "Input datatype must be a floating point type, got %s", first);
        return Collections.singletonList(first);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
