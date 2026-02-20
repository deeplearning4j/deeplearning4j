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

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * SAM-style Two-Way Cross Attention.
 *
 * Bidirectional cross-attention where:
 *   1. tokenOutput = softmax(tokenQ @ imageK^T * scale) @ imageV
 *   2. imageOutput = softmax(imageQ @ tokenK^T * scale) @ tokenV
 *
 * Inputs:
 *   0: tokenQuery  [batch, tokenSeqLen, embedDim]
 *   1: tokenKey    [batch, tokenSeqLen, embedDim]
 *   2: tokenValue  [batch, tokenSeqLen, embedDim]
 *   3: imageQuery  [batch, imageSeqLen, embedDim]
 *   4: imageKey    [batch, imageSeqLen, embedDim]
 *   5: imageValue  [batch, imageSeqLen, embedDim]
 *
 * Float args:
 *   0: scale factor (default: 1/sqrt(embedDim))
 *
 * Output:
 *   0: tokenOutput [batch, tokenSeqLen, embedDim]
 *   1: imageOutput [batch, imageSeqLen, embedDim]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class TwoWayCrossAttention extends DynamicCustomOp {

    @Getter private double scale = 0.0;  // 0 means auto = 1/sqrt(dim)

    public TwoWayCrossAttention(INDArray tokenQuery, INDArray tokenKey, INDArray tokenValue,
                                 INDArray imageQuery, INDArray imageKey, INDArray imageValue) {
        super(new INDArray[]{tokenQuery, tokenKey, tokenValue, imageQuery, imageKey, imageValue}, null);
    }

    public TwoWayCrossAttention(INDArray tokenQuery, INDArray tokenKey, INDArray tokenValue,
                                 INDArray imageQuery, INDArray imageKey, INDArray imageValue,
                                 double scale) {
        super(new INDArray[]{tokenQuery, tokenKey, tokenValue, imageQuery, imageKey, imageValue}, null);
        this.scale = scale;
        if (scale > 0) addTArgument(scale);
    }

    public TwoWayCrossAttention(SameDiff sameDiff, SDVariable tokenQuery, SDVariable tokenKey,
                                 SDVariable tokenValue, SDVariable imageQuery, SDVariable imageKey,
                                 SDVariable imageValue) {
        super(null, sameDiff, new SDVariable[]{tokenQuery, tokenKey, tokenValue,
                imageQuery, imageKey, imageValue}, false);
    }

    public TwoWayCrossAttention(SameDiff sameDiff, SDVariable tokenQuery, SDVariable tokenKey,
                                 SDVariable tokenValue, SDVariable imageQuery, SDVariable imageKey,
                                 SDVariable imageValue, double scale) {
        super(null, sameDiff, new SDVariable[]{tokenQuery, tokenKey, tokenValue,
                imageQuery, imageKey, imageValue}, false);
        this.scale = scale;
        if (scale > 0) addTArgument(scale);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.scale = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "two_way_cross_attention";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Arrays.asList(new TwoWayCrossAttentionBp(sameDiff,
                arg(0), arg(1), arg(2), arg(3), arg(4), arg(5),
                grad.get(0), grad.get(1), scale).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Arrays.asList(inputDataTypes.get(0), inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
