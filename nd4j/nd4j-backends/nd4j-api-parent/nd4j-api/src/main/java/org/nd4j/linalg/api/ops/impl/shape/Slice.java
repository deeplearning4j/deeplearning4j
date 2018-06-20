/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Slice function
 *
 * @author Adam Gibson
 */
@Slf4j
public class Slice extends DynamicCustomOp {

    private int[] begin;
    private int[] size;

    public Slice() {}

    public Slice(SameDiff sameDiff, @NonNull SDVariable input, @NonNull int[] begin, @NonNull int[] size){
        super(null, sameDiff, new SDVariable[]{input});
        this.begin = begin;
        this.size = size;
        addIArgument(begin);
        addIArgument(size);
    }


    @Override
    public String opName() {
        return "slice";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Slice";
    }


    @Override
    public void assertValidForExecution() {
        if(numInputArguments() != 1 && numInputArguments() != 3 && numInputArguments() != 4) {
            throw new ND4JIllegalStateException("Num input arguments must be 1 3 or 4.");
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
         /*
            strided slice typically takes 4 tensor arguments:
            0) input, it's shape determines number of elements in other arguments
            1) begin indices
            2) end indices
            3) strides
         */

        val inputBegin = nodeDef.getInput(1);
        val inputEnd = nodeDef.getInput(2);

        NodeDef beginNode = null;
        NodeDef endNode = null;

        for(int i = 0; i < graph.getNodeCount(); i++) {
            if(graph.getNode(i).getName().equals(inputBegin)) {
                beginNode = graph.getNode(i);
            }
            if(graph.getNode(i).getName().equals(inputEnd)) {
                endNode = graph.getNode(i);
            }

        }



        val beginArr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",beginNode,graph);
        val endArr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",endNode,graph);

        if (beginArr != null && endArr != null) {

            for (int e = 0; e < beginArr.length(); e++)
                addIArgument(beginArr.getInt(e));

            for (int e = 0; e <  endArr.length(); e++)
                addIArgument(endArr.getInt(e));


        } else {
            // do nothing
        }



    }



    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val beginMapping = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"begin"})
                .build();

        val size = PropertyMapping.builder()
                .tfInputPosition(2)
                .propertyNames(new String[]{"size"})
                .build();





        map.put("begin",beginMapping);
        map.put("size",size);



        ret.put(tensorflowName(),map);

        return ret;
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        return super.propertiesForFunction();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Collections.singletonList(f().sliceBp(arg(), grad.get(0), begin, size));
    }
}
