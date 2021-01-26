/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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
package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class FakeQuantWithMinMaxVarsPerChannel extends DynamicCustomOp {
    protected boolean narrowRange;
    protected int numBits;

    public FakeQuantWithMinMaxVarsPerChannel() {}

    public FakeQuantWithMinMaxVarsPerChannel(INDArray x, INDArray min, INDArray max, int num_bits, boolean narrow) {
        Preconditions.checkArgument(min.isVector() && max.isVector() &&
                        min.length() == max.length(),
                "FakeQuantWithMinMaxVarsPerChannel: min and max should be 1D tensors with the same length");
        addInputArgument(x,min,max);
        addIArgument(num_bits);
        addBArgument(narrow);
    }

    public FakeQuantWithMinMaxVarsPerChannel(INDArray x, INDArray min, INDArray max, int num_bits) {
        this(x, min, max, num_bits, false);
    }

    public FakeQuantWithMinMaxVarsPerChannel(INDArray x, INDArray min, INDArray max, boolean narrow) {
        this(x, min, max, 8, narrow);
    }

    public FakeQuantWithMinMaxVarsPerChannel(INDArray x, INDArray min, INDArray max) {
        this(x, min, max, 8, false);
    }

    public FakeQuantWithMinMaxVarsPerChannel(SameDiff sameDiff, SDVariable x, SDVariable min, SDVariable max,
                                             int num_bits, boolean narrow) {
        super("", sameDiff, new SDVariable[]{x, min, max});
        addIArgument(num_bits);
        addBArgument(narrow);
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
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(attributesForNode.containsKey("narrow_range")){
            this.narrowRange = attributesForNode.get("narrow_range").getB();
        }
        if(attributesForNode.containsKey("num_bits")) {
            this.numBits = (int) attributesForNode.get("num_bits").getI();
        }
        addIArgument(numBits);
        addBArgument(narrowRange);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected exactly 3 inputs, got %s", inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}