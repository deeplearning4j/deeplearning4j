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
 * Backward pass for MeanSquare operation.
 *
 * Computes gradient for mean_square(x) = mean(x * x)
 * Gradient: d/dx[mean(x^2)] = 2*x / n
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class MeanSquareBp extends DynamicCustomOp {

    @Getter private boolean keepDims = true;

    /**
     * Create a mean_square_bp operation for SameDiff.
     *
     * @param sameDiff The SameDiff instance
     * @param input Input tensor
     * @param gradOut Gradient of the output
     * @param keepDims Whether dimensions were kept in the forward pass
     */
    public MeanSquareBp(SameDiff sameDiff, SDVariable input, SDVariable gradOut, boolean keepDims) {
        super(null, sameDiff, new SDVariable[]{input, gradOut}, false);
        this.keepDims = keepDims;
        addIArgument(keepDims ? 1 : 0);
    }

    /**
     * Create a mean_square_bp operation for INDArray execution.
     *
     * @param input Input array
     * @param gradOut Gradient of the output
     * @param gradIn Gradient w.r.t. input (output)
     * @param keepDims Whether dimensions were kept in the forward pass
     */
    public MeanSquareBp(INDArray input, INDArray gradOut, INDArray gradIn, boolean keepDims) {
        super(new INDArray[]{input, gradOut}, gradIn != null ? new INDArray[]{gradIn} : null);
        this.keepDims = keepDims;
        addIArgument(keepDims ? 1 : 0);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.keepDims = iArguments.get(0) != 0;
        }
    }

    @Override
    public String opName() {
        return "mean_square_bp";
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
