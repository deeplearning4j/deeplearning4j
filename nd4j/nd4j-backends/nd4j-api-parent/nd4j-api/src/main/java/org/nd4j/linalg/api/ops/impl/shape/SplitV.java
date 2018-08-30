/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * SplitV op
 */
public class SplitV extends DynamicCustomOp {

    private int numSplit;
    private int splitDim;

    @Override
    public String opName() {
        return "split_v";
    }

    @Override
    public String tensorflowName() {
        return "SplitV";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val splitDim = TFGraphMapper.getInstance().getArrayFrom(TFGraphMapper.getInstance().getNodeWithNameFromGraph(graph,nodeDef.getInput(0)),graph);
        if(splitDim != null) {
            this.splitDim = splitDim.getInt(0);
            addIArgument(splitDim.getInt(0));
        }

        //numSplits is sometimes available for import, but libnd4j op doesn't used/need it for execution
        //val numSplits = (int) attributesForNode.get("num_split").getI();
        //this.numSplit = numSplits;
        //addIArgument(numSplits);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("numSplit",numSplit);
        ret.put("splitDim",splitDim);
        return ret;
    }



    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val splitDim = PropertyMapping.builder()
                .tfInputPosition(-1)
                .propertyNames(new String[]{"splitDim"})
                .build();

        val numSplit = PropertyMapping.builder()
                .tfAttrName("num_split")
                .propertyNames(new String[]{"numSplit"})
                .build();

        map.put("numSplit",numSplit);
        map.put("splitDim",splitDim);

        ret.put(tensorflowName(),map);
        //ret.put(onnxName(),map);

        return ret;
    }

    @Override
    public int getNumOutputs(){
        return numSplit;
    }

}
