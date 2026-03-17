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
 * Fused Bias + Dropout + Residual
 *
 * Computes: dropout(input + bias) + residual in a single kernel
 * to minimize memory bandwidth usage.
 *
 * This is a common pattern in transformer attention and FFN blocks.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FusedBiasDropoutResidual extends DynamicCustomOp {

    @Getter private double dropoutProb = 0.0;
    @Getter private long seed = 0;
    @Getter private boolean training = false;

    /**
     * Create a fused bias + dropout + residual operation.
     *
     * @param sameDiff The SameDiff instance
     * @param input Input tensor
     * @param bias Bias tensor (broadcastable to input)
     * @param residual Residual connection tensor
     * @param dropoutProb Dropout probability (0 = no dropout)
     * @param seed Random seed for dropout
     * @param training Whether in training mode
     */
    public FusedBiasDropoutResidual(SameDiff sameDiff, SDVariable input, SDVariable bias, SDVariable residual,
                                    double dropoutProb, long seed, boolean training) {
        super(null, sameDiff, new SDVariable[]{input, bias, residual}, false);
        this.dropoutProb = dropoutProb;
        this.seed = seed;
        this.training = training;

        addIArgument(seed);
        addTArgument(dropoutProb);
        addBArgument(training);
    }

    public FusedBiasDropoutResidual(SameDiff sameDiff, SDVariable input, SDVariable bias, SDVariable residual) {
        this(sameDiff, input, bias, residual, 0.0, 0, false);
    }

    public FusedBiasDropoutResidual(SameDiff sd, SDVariable input, SDVariable bias, SDVariable residual,
                                    double dropoutProb, boolean training) {
        this(sd, input, bias, residual, dropoutProb, 0, training);
    }

    public FusedBiasDropoutResidual(INDArray input, INDArray bias, INDArray residual,
                                    double dropoutProb, boolean training) {
        this(input, bias, residual, null, dropoutProb, 0, training);
    }

    public FusedBiasDropoutResidual(INDArray input, INDArray bias, INDArray residual, INDArray output,
                                    double dropoutProb, long seed, boolean training) {
        super(new INDArray[]{input, bias, residual}, output != null ? new INDArray[]{output} : null);
        this.dropoutProb = dropoutProb;
        this.seed = seed;
        this.training = training;

        addIArgument(seed);
        addTArgument(dropoutProb);
        addBArgument(training);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.seed = iArguments.get(0);
        }
        if (tArguments.size() > 0) {
            this.dropoutProb = tArguments.get(0);
        }
        if (bArguments.size() > 0) {
            this.training = bArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "fused_bias_dropout_residual";
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
