/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

import java.util.Collections;
import java.util.List;
import lombok.Getter;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.ThresholdReluBp;

/**
 * Threshold ReLU op.  The genral case of {@link RectifiedLinear}.
 */
public class ThresholdRelu extends DynamicCustomOp {

    @Getter
    private double cutoff = 0.0;

    public ThresholdRelu(){ }

    public ThresholdRelu(SameDiff sd, SDVariable input, boolean inPlace, double cutoff){
        super(sd, new SDVariable[]{input}, inPlace);
        this.cutoff = cutoff;
        addTArgument(cutoff);
    }

    public ThresholdRelu(SameDiff sd, SDVariable input, double cutoff){
        super(sd, new SDVariable[]{input});
        this.cutoff = cutoff;
        addTArgument(cutoff);
    }

    public ThresholdRelu(@NonNull INDArray input, INDArray output, double cutoff){
        super(new INDArray[]{input}, wrapOrNull(output));
        this.cutoff = cutoff;
        addTArgument(cutoff);
    }

    @Override
    public String opName(){
        return "thresholdedrelu";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions
                .checkArgument(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatypes, got %s", dataTypes);
        Preconditions.checkArgument(dataTypes.get(0).isFPType(), "Input datatype must be floating point, got %s", dataTypes);

        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return new ThresholdReluBp(sameDiff, arg(), f1.get(0), cutoff).outputs();
    }
}
