/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class FakeQuantWithMinMaxVarsPerChannel extends DynamicCustomOp {
    public FakeQuantWithMinMaxVarsPerChannel() {}

    public FakeQuantWithMinMaxVarsPerChannel(INDArray x, INDArray min, INDArray max) {
        Preconditions.checkArgument(min.isVector() && max.isVector() &&
                        min.length() == max.length(),
                "FakeQuantWithMinMaxVarsPerChannel: min and max should be 1D tensors with the same length");
        inputArguments.add(x);
        inputArguments.add(min);
        inputArguments.add(max);
    }

    public FakeQuantWithMinMaxVarsPerChannel(INDArray x, INDArray min, INDArray max,
                                             INDArray output) {
        this(x,min,max);
        outputArguments.add(output);
    }

    public FakeQuantWithMinMaxVarsPerChannel(SameDiff sameDiff, SDVariable x, SDVariable min, SDVariable max) {
        super("", sameDiff, new SDVariable[]{x, min, max});
    }

    @Override
    public String opName() {
        return "fake_quant_with_min_max_vars_per_channel";
    }

    @Override
    public String tensorflowName() {
        return "FakeQuantWithMinMaxVarsPerChannel";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected exactly 3 inputs, got %s", inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}