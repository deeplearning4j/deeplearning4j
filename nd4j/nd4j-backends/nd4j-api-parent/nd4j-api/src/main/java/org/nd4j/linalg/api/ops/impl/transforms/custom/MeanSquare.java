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
 * Mean of Squared Values Operation
 *
 * Computes mean(x * x) along the last dimension.
 * This is a building block for RMSNorm and other normalization operations.
 *
 * RMSNorm uses this as: rsqrt(mean_square(x) + epsilon) * x * gamma
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class MeanSquare extends DynamicCustomOp {

    @Getter private boolean keepDims = true;

    /**
     * Create a mean_square operation for SameDiff.
     *
     * @param sameDiff The SameDiff instance
     * @param input Input tensor
     * @param keepDims Whether to keep the reduced dimension
     */
    public MeanSquare(SameDiff sameDiff, SDVariable input, boolean keepDims) {
        super(null, sameDiff, new SDVariable[]{input}, false);
        this.keepDims = keepDims;
        addIArgument(keepDims ? 1 : 0);
    }

    /**
     * Create a mean_square operation for SameDiff with keepDims=true.
     *
     * @param sameDiff The SameDiff instance
     * @param input Input tensor
     */
    public MeanSquare(SameDiff sameDiff, SDVariable input) {
        this(sameDiff, input, true);
    }

    /**
     * Create a mean_square operation for INDArray execution.
     *
     * @param input Input array
     * @param output Output array (optional)
     * @param keepDims Whether to keep the reduced dimension
     */
    public MeanSquare(INDArray input, INDArray output, boolean keepDims) {
        super(new INDArray[]{input}, output != null ? new INDArray[]{output} : null);
        this.keepDims = keepDims;
        addIArgument(keepDims ? 1 : 0);
    }

    /**
     * Create a mean_square operation for INDArray execution with keepDims=true.
     *
     * @param input Input array
     */
    public MeanSquare(INDArray input) {
        this(input, null, true);
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
        return "mean_square";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        return Arrays.asList(new MeanSquareBp(sameDiff, arg(0), gradients.get(0), keepDims).outputVariables());
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
