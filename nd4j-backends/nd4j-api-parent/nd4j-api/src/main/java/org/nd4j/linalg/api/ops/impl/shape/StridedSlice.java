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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Reshape function
 *
 * @author Adam Gibson
 */
@Slf4j
public class StridedSlice extends DynamicCustomOp {


    public StridedSlice() {}


    @Override
    public String opName() {
        return "stridedslice";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "StridedSlice";
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
        val inputStrides = nodeDef.getInput(3);

        NodeDef beginNode = null;
        NodeDef endNode = null;
        NodeDef strides = null;

        for(int i = 0; i < graph.getNodeCount(); i++) {
            if(graph.getNode(i).getName().equals(inputBegin)) {
                beginNode = graph.getNode(i);
            }
            if(graph.getNode(i).getName().equals(inputEnd)) {
                endNode = graph.getNode(i);
            }
            if(graph.getNode(i).getName().equals(inputStrides)) {
                strides = graph.getNode(i);
            }
        }


        // bit masks for this slice
        val bm = nodeDef.getAttrOrThrow("begin_mask");
        val xm = nodeDef.getAttrOrThrow("ellipsis_mask");
        val em = nodeDef.getAttrOrThrow("end_mask");
        val nm = nodeDef.getAttrOrThrow("new_axis_mask");
        val sm = nodeDef.getAttrOrThrow("shrink_axis_mask");

        addIArgument((int) bm.getI());
        addIArgument((int) xm.getI());
        addIArgument((int) em.getI());

        addIArgument((int) nm.getI());
        addIArgument((int) sm.getI());

        val beginArr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",beginNode,graph);
        val endArr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",endNode,graph);
        val stridesArr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",strides,graph);

        if (beginArr != null && endArr != null && stridesArr != null) {

            for (int e = 0; e < beginArr.length(); e++)
                addIArgument(beginArr.getInt(e));

            for (int e = 0; e <  endArr.length(); e++)
                addIArgument(endArr.getInt(e));

            for (int e = 0; e < stridesArr.length(); e++)
                addIArgument(stridesArr.getInt(e));
        } else {
            // do nothing
        }



    }


    @Override
    public void initWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret = outputVariables()[0];

        return Collections.singletonList(ret);
    }

}
