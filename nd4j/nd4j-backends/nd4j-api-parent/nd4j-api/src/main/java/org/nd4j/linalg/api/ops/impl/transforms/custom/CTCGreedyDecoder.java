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
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * CTC Greedy Decoder - Connectionist Temporal Classification decoding
 *
 * Performs greedy decoding on CTC output (log probabilities).
 * Used in OCR and speech recognition to decode CTC output to text sequences.
 *
 * @author Eclipse Deeplearning4j
 */
@NoArgsConstructor
@Getter
public class CTCGreedyDecoder extends DynamicCustomOp {

    private boolean mergeRepeated = true;
    private int blankIndex = 0;

    /**
     * Create a CTC greedy decoder operation.
     *
     * @param logits Log probabilities [batch, time, num_classes]
     * @param mergeRepeated Whether to merge repeated characters (default true)
     * @param blankIndex Index of blank label (default 0)
     */
    public CTCGreedyDecoder(@NonNull INDArray logits, boolean mergeRepeated, int blankIndex) {
        super(new INDArray[]{logits}, null);
        this.mergeRepeated = mergeRepeated;
        this.blankIndex = blankIndex;
        addArgs();
    }

    /**
     * Create a CTC greedy decoder operation with sequence lengths.
     *
     * @param logits Log probabilities [batch, time, num_classes]
     * @param sequenceLength Actual sequence lengths [batch]
     * @param mergeRepeated Whether to merge repeated characters (default true)
     * @param blankIndex Index of blank label (default 0)
     */
    public CTCGreedyDecoder(@NonNull INDArray logits, INDArray sequenceLength,
                           boolean mergeRepeated, int blankIndex) {
        super(sequenceLength != null ? new INDArray[]{logits, sequenceLength} : new INDArray[]{logits}, null);
        this.mergeRepeated = mergeRepeated;
        this.blankIndex = blankIndex;
        addArgs();
    }

    /**
     * Create a CTC greedy decoder for SameDiff.
     *
     * @param sd SameDiff instance
     * @param logits Log probabilities [batch, time, num_classes]
     * @param mergeRepeated Whether to merge repeated characters
     * @param blankIndex Index of blank label
     */
    public CTCGreedyDecoder(@NonNull SameDiff sd, @NonNull SDVariable logits,
                           boolean mergeRepeated, int blankIndex) {
        super(null, sd, new SDVariable[]{logits}, false);
        this.mergeRepeated = mergeRepeated;
        this.blankIndex = blankIndex;
        addArgs();
    }

    /**
     * Create a CTC greedy decoder for SameDiff with sequence lengths.
     *
     * @param sd SameDiff instance
     * @param logits Log probabilities [batch, time, num_classes]
     * @param sequenceLength Actual sequence lengths [batch]
     * @param mergeRepeated Whether to merge repeated characters
     * @param blankIndex Index of blank label
     */
    public CTCGreedyDecoder(@NonNull SameDiff sd, @NonNull SDVariable logits,
                           SDVariable sequenceLength, boolean mergeRepeated, int blankIndex) {
        super(null, sd, sequenceLength != null ? new SDVariable[]{logits, sequenceLength} : new SDVariable[]{logits}, false);
        this.mergeRepeated = mergeRepeated;
        this.blankIndex = blankIndex;
        addArgs();
    }

    private void addArgs() {
        addIArgument(mergeRepeated ? 1 : 0, blankIndex);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.mergeRepeated = iArguments.get(0) != 0;
        }
        if (iArguments.size() > 1) {
            this.blankIndex = iArguments.get(1).intValue();
        }
    }

    @Override
    public String opName() {
        return "ctc_greedy_decoder";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 1,
                "Expected at least 1 input data type for %s, got %s", getClass(), inputDataTypes);
        // Output 0: decoded [batch, time] - INT64
        // Output 1: log_probability [batch] - same as input
        return Arrays.asList(DataType.INT64, inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }

    @Override
    public String onnxName() {
        return "CTCGreedyDecoder";
    }
}
