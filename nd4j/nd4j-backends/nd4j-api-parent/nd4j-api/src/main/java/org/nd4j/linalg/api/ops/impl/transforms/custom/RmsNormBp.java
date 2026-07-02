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
 * Backpropagation for RmsNorm.
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class RmsNormBp extends DynamicCustomOp {

    @Getter private double epsilon = 1e-5;

    public RmsNormBp(SameDiff sameDiff, SDVariable input, SDVariable gradOut, double epsilon) {
        super(null, sameDiff, new SDVariable[]{input, gradOut}, false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    /**
     * Constructor with optional gamma (scale weights). When gamma is non-null a second
     * output (dL/dgamma) is produced in addition to the input gradient.
     */
    public RmsNormBp(SameDiff sameDiff, SDVariable input, SDVariable gradOut, SDVariable gamma, double epsilon) {
        super(null, sameDiff,
              gamma != null ? new SDVariable[]{input, gradOut, gamma} : new SDVariable[]{input, gradOut},
              false);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public RmsNormBp(INDArray input, INDArray gradOut, INDArray gradIn, double epsilon) {
        super(new INDArray[]{input, gradOut}, gradIn != null ? new INDArray[]{gradIn} : null);
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    public RmsNormBp(INDArray input, INDArray gradOut, INDArray gamma, INDArray gradIn, INDArray gradGamma, double epsilon) {
        super(gamma != null ? new INDArray[]{input, gradOut, gamma} : new INDArray[]{input, gradOut},
              buildOutputs(gradIn, gradGamma));
        this.epsilon = epsilon;
        addTArgument(epsilon);
    }

    private static INDArray[] buildOutputs(INDArray gradIn, INDArray gradGamma) {
        if (gradGamma != null) return new INDArray[]{gradIn, gradGamma};
        if (gradIn != null) return new INDArray[]{gradIn};
        return null;
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!tArguments.isEmpty()) {
            this.epsilon = tArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "rms_norm_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        // inputs: [input, gradOut] or [input, gradOut, gamma]
        // outputs: [gradInput] or [gradInput, gradGamma]
        if (inputDataTypes.size() >= 3) {
            // gamma is present — output both gradInput and gradGamma
            return Arrays.asList(inputDataTypes.get(0), inputDataTypes.get(2));
        }
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        // When gamma (input index 2) is present, we output an extra gradient for it.
        if (args() != null && args().length >= 3) return 2;
        if (inputArguments() != null && inputArguments().size() >= 3) return 2;
        return 1;
    }
}
