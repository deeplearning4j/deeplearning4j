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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NonNull;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Split op
 */
public class Split extends DynamicCustomOp {

    private int numSplit;
    private int splitDim;

    public Split() {
    }

    public Split(SameDiff sameDiff, SDVariable input, int numSplit, int splitDim) {
        super(null,sameDiff,new SDVariable[]{input});
        this.numSplit = numSplit;
        this.splitDim = splitDim;
        addIArgument(numSplit,splitDim);
    }

    public Split(@NonNull INDArray in, INDArray out) {
        super(null, new INDArray[]{in}, wrapOrNull(out), null, (List<Integer>)null);
    }


    @Override
    public String opName() {
        return "split";
    }

    @Override
    public String tensorflowName() {
        return "Split";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val numSplits = (int) attributesForNode.get("num_split").getI();
        this.numSplit = numSplits;
        addIArgument(numSplits);

        val splitDim = TFGraphMapper.getArrayFrom(TFGraphMapper.getNodeWithNameFromGraph(graph,nodeDef.getInput(0)),graph);
        if(splitDim != null) {
            this.splitDim = splitDim.getInt(0);
            addIArgument(splitDim.getInt(0));
        }
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val splitDim = PropertyMapping.builder()
                .tfInputPosition(0)
                .propertyNames(new String[]{"splitDim"})
                .build();

        val numSplit = PropertyMapping.builder()
                .tfAttrName("num_split")
                .propertyNames(new String[]{"numSplit"})
                .build();

        map.put("numSplit",numSplit);
        map.put("splitDim",splitDim);

        ret.put(tensorflowName(),map);

        return ret;
    }

    @Override
    public int getNumOutputs(){
        return numSplit;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && !dataTypes.isEmpty(), "No datatypes were provided for %s: %s", getClass(), dataTypes);
        DataType dt;
        if(dataTypes.size() == 1) {
            dt = dataTypes.get(0);
        } else {
            //Order seems to usually be axis first for TF import? libnd4j supports both...
            if(dataTypes.get(0).isIntType()){
                dt = dataTypes.get(1);
            } else {
                dt = dataTypes.get(0);
            }
        }
        //Output types are same as first input type - just numSplits of them...
        List<DataType> out = new ArrayList<>(numSplit);
        for( int i = 0; i < numSplit; i++) {
            out.add(dt);
        }
        return out;
    }
}
