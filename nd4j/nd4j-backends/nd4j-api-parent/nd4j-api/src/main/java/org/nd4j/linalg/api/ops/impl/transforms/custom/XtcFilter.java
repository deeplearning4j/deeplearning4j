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
 * Exclude Top Choices (XTC) logit filter.
 *
 * With probability xtcProbability: among tokens whose softmax probability >= xtcThreshold,
 * if at least two qualify, mask all EXCEPT the lowest-probability one (encouraging diversity).
 * With probability (1 - xtcProbability) the logits are returned unchanged. Stochastic:
 * the apply/skip draw uses a native RNG seeded by {@code seed}. Masked positions are -inf.
 *
 * Input:
 *   0: logits [vocabSize] or [batch, vocabSize]
 *
 * Output:
 *   0: filtered logits (same shape and type as input)
 *
 * Float args:
 *   0: xtcProbability (0.0 = off)
 *   1: xtcThreshold — per-token probability threshold (must be < 0.5)
 *
 * Int args:
 *   0: seed
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class XtcFilter extends DynamicCustomOp {

    @Getter private double xtcProbability = 0.0;
    @Getter private double xtcThreshold = 0.1;
    @Getter private long seed = 0;

    public XtcFilter(INDArray logits, double xtcProbability, double xtcThreshold, long seed) {
        super(new INDArray[]{logits}, null);
        this.xtcProbability = xtcProbability;
        this.xtcThreshold = xtcThreshold;
        this.seed = seed;
        addTArgument(xtcProbability, xtcThreshold);
        addIArgument(seed);
    }

    public XtcFilter(SameDiff sameDiff, SDVariable logits, double xtcProbability, double xtcThreshold, long seed) {
        super(null, sameDiff, new SDVariable[]{logits}, false);
        this.xtcProbability = xtcProbability;
        this.xtcThreshold = xtcThreshold;
        this.seed = seed;
        addTArgument(xtcProbability, xtcThreshold);
        addIArgument(seed);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.xtcProbability = tArguments.get(0);
        if (tArguments.size() > 1) this.xtcThreshold = tArguments.get(1);
        if (iArguments.size() > 0) this.seed = iArguments.get(0);
    }

    @Override
    public String opName() {
        return "xtc_filter";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
