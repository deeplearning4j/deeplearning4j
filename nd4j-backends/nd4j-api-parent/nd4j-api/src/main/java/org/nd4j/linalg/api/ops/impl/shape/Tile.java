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

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Tile function
 *
 * @author Adam Gibson
 */
public class Tile extends DynamicCustomOp {

    private int[] axis;
    private boolean is_static_reps = false;

    public Tile(SameDiff sameDiff, SDVariable i_v, int[] axis) {
        super(null,sameDiff, new SDVariable[]{i_v}, false);
        this.axis = axis;
        addArguments();
    }

    public Tile(INDArray[] inputs, INDArray[] outputs, int[] axis, boolean is_static_reps) {
        super(null, inputs, outputs);
        this.axis = axis;
        this.is_static_reps = is_static_reps;
        addArguments();
    }


    public Tile(INDArray[] inputs, INDArray[] outputs, int[] axis) {
        this(inputs,outputs,axis,false);
    }


    public Tile() {}

    private void addArguments() {
        this.is_static_reps = true;
        addIArgument(axis);
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val lastNode = TFGraphMapper.getInstance().getNodeWithNameFromGraph(graph,nodeDef.getInput(nodeDef.getInputCount() - 1));
        val arr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",lastNode,graph);
        if(arr != null) {
            this.axis = arr.data().asInt();
            addArguments();
        }
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("axis",axis);
        return ret;
    }


    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        populateInputsAndOutputsFromSameDiff();
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val axisMapping = PropertyMapping.builder()
                .onnxAttrName("axis")
                .tfInputPosition(-1)
                .propertyNames(new String[]{"axis"})
                .build();

        map.put("axis",axisMapping);

        ret.put(tensorflowName(),map);
        ret.put(onnxName(),map);

        return ret;
    }

    @Override
    public List<int[]> calculateOutputShape() {
        /**
         * This op is special case: we can't infer its shape before both inputs are available.
         * So if reps argument is full of 0.0s - we skip shape inference
         *
         * And during actual op invocation both inputs should be available due to topo sort
         */
        if (is_static_reps)
            return Nd4j.getExecutioner().calculateOutputShape(this);

        if (inputArguments().length < 2)
            return Collections.emptyList();

        val array = inputArguments()[1];
        val reps = new int[array.length()];

        for (int e = 0; e < reps.length; e++)
            reps[e] = (int) array.getDouble(e);

        if (ArrayUtil.prod(reps) == 0)
            return Collections.emptyList();
        else
            return Nd4j.getExecutioner().calculateOutputShape(this);
    }


    @Override
    public String opName() {
        return "tile";
    }

    @Override
    public String onnxName() {
        return "Tile";
    }

    @Override
    public String tensorflowName() {
        return "Tile";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException();
    }

}
