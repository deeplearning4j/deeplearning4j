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

import java.util.Collections;
import java.util.List;

/**
 * Token sampling for LLM inference.
 *
 * Full sampling pipeline in a single native call:
 *   temperature scaling -> top-K filtering -> softmax -> top-P filtering -> sample/argmax
 *
 * For greedy decoding (temperature=0 or no top-k/top-p), performs GPU-side argmax
 * with shared-memory reduction — avoids transferring the full logits tensor to host.
 *
 * Inputs:
 *   0: logits [batch, vocabSize] or [vocabSize]
 *
 * Output:
 *   0: token indices INT64 [batch] or scalar
 *
 * Float args:
 *   0: temperature (default: 0.0 = greedy/argmax)
 *   1: topP nucleus threshold (default: 0.0 = disabled)
 *
 * Int args:
 *   0: topK (default: 0 = disabled)
 *   1: seed (default: 0 = random)
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class TokenSample extends DynamicCustomOp {

    @Getter private double temperature = 0.0;
    @Getter private double topP = 0.0;
    @Getter private int topK = 0;
    @Getter private long seed = 0;

    /**
     * Greedy sampling (argmax).
     */
    public TokenSample(INDArray logits) {
        this(logits, 0.0, 0, 0.0, 0);
    }

    /**
     * Full sampling with all parameters.
     */
    public TokenSample(INDArray logits, double temperature, int topK, double topP, long seed) {
        super(new INDArray[]{logits}, null);
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.seed = seed;
        addTArgument(temperature, topP);
        addIArgument((long) topK, seed);
    }

    /**
     * SameDiff constructor — greedy.
     */
    public TokenSample(SameDiff sameDiff, SDVariable logits) {
        this(sameDiff, logits, 0.0, 0, 0.0, 0);
    }

    /**
     * SameDiff constructor — full parameters.
     */
    public TokenSample(SameDiff sameDiff, SDVariable logits, double temperature, int topK, double topP, long seed) {
        super(null, sameDiff, new SDVariable[]{logits}, false);
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.seed = seed;
        addTArgument(temperature, topP);
        addIArgument((long) topK, seed);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.temperature = tArguments.get(0);
        if (tArguments.size() > 1) this.topP = tArguments.get(1);
        if (iArguments.size() > 0) this.topK = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.seed = iArguments.get(1);
    }

    @Override
    public String opName() {
        return "token_sample";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(DataType.INT64);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
