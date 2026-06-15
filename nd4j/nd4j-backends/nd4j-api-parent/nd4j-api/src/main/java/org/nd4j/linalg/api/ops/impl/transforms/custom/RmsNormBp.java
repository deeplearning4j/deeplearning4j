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

    public RmsNormBp(INDArray input, INDArray gradOut, INDArray gradIn, double epsilon) {
        super(new INDArray[]{input, gradOut}, gradIn != null ? new INDArray[]{gradIn} : null);
        this.epsilon = epsilon;
        addTArgument(epsilon);
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
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
