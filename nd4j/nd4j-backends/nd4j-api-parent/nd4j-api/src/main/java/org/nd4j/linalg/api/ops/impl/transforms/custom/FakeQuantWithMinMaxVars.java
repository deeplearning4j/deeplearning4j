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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class FakeQuantWithMinMaxVars extends DynamicCustomOp {

    protected boolean narrowRange;
    protected int numBits;

    public FakeQuantWithMinMaxVars(SameDiff sd, SDVariable input, SDVariable min, SDVariable max, boolean narrowRange, int numBits){
        super(sd, new SDVariable[]{input, min, max});
        Preconditions.checkState(numBits >= 2 && numBits <= 16, "NumBits arg must be in range 2 to 16 inclusive, got %s", numBits);
        this.narrowRange = narrowRange;
        this.numBits = numBits;
        addArgs();
    }

    public FakeQuantWithMinMaxVars(INDArray x, INDArray min, INDArray max, int num_bits, boolean narrow) {
        Preconditions.checkArgument(min.isVector() && max.isVector() &&
                        min.length() == max.length(),
                "FakeQuantWithMinMaxVars: min and max should be 1D tensors with the same length");
        addInputArgument(x,min,max);
        addIArgument(num_bits);
        addBArgument(narrow);
    }

    public FakeQuantWithMinMaxVars(){ }

    protected void addArgs(){
        iArguments.clear();
        bArguments.clear();
        addIArgument(numBits);
        addBArgument(narrowRange);
    }

    @Override
    public String opName(){
        return "fake_quant_with_min_max_vars";
    }

    @Override
    public String tensorflowName(){
        return "FakeQuantWithMinMaxVars";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if(attributesForNode.containsKey("narrow_range")){
            this.narrowRange = attributesForNode.get("narrow_range").getB();
        }
        this.numBits = (int)attributesForNode.get("num_bits").getI();
        addArgs();
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected exactly 3 inputs, got %s", inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Arrays.asList(sameDiff.zerosLike(arg(0)), sameDiff.zerosLike(arg(1)), sameDiff.zerosLike(arg(2)));
    }
}
