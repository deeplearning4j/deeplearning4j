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

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Strided Slice function
 *
 * @author Adam Gibson
 */
@Slf4j
public class StridedSlice extends DynamicCustomOp {
    private int[] begin;
    private int[] end;
    private int[] strides;
    private int beginMask;
    private int endMask;
    private int ellipsisMask;
    private int newAxisMask;
    private int shrinkAxisMask;

    public StridedSlice() {}

    public StridedSlice(SameDiff sameDiff, SDVariable in, int[] begin, int[] end, int[] strides){
        this(sameDiff, in, begin, end, strides, 0, 0, 0, 0, 0);
    }

    public StridedSlice(SameDiff sameDiff, SDVariable in, @NonNull int[] begin, @NonNull int[] end, @NonNull int[] strides,
                        int beginMask, int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask){
        super(null, sameDiff, new SDVariable[]{in});
        this.begin = begin;
        this.end = end;
        this.strides = strides;
        this.beginMask = beginMask;
        this.endMask = endMask;
        this.ellipsisMask = ellipsisMask;
        this.newAxisMask = newAxisMask;
        this.shrinkAxisMask = shrinkAxisMask;

        //https://github.com/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/parity_ops/strided_slice.cpp#L279
        addIArgument(beginMask);
        addIArgument(ellipsisMask);
        addIArgument(endMask);
        addIArgument(newAxisMask);
        addIArgument(shrinkAxisMask);
        addIArgument(begin);
        addIArgument(end);
        addIArgument(strides);
    }


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
    public void assertValidForExecution() {
        if(numInputArguments() != 1 && numInputArguments() != 3 && numInputArguments() != 4) {
            throw new ND4JIllegalStateException("Num input arguments must be 1 3 or 4.");
        }

        if(numIArguments() < 5) {
            throw new ND4JIllegalStateException("Number of integer arguments must >= 5");
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
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
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val beginMapping = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"begin"})
                .build();

        val end = PropertyMapping.builder()
                .tfInputPosition(2)
                .propertyNames(new String[]{"end"})
                .build();


        val strides = PropertyMapping.builder()
                .tfInputPosition(3)
                .propertyNames(new String[]{"strides"})
                .build();




        val beginMask = PropertyMapping.builder()
                .tfAttrName("begin_mask")
                .propertyNames(new String[]{"beginMask"})
                .build();


        val ellipsisMask = PropertyMapping.builder()
                .tfAttrName("ellipsis_mask")
                .propertyNames(new String[]{"ellipsisMask"})
                .build();



        val endMask = PropertyMapping.builder()
                .tfAttrName("end_mask")
                .propertyNames(new String[]{"endMask"})
                .build();



        val newAxisMask = PropertyMapping.builder()
                .tfAttrName("new_axis_mask")
                .propertyNames(new String[]{"newAxisMask"})
                .build();

        val shrinkAxisMask = PropertyMapping.builder()
                .tfAttrName("shrink_axis_mask")
                .propertyNames(new String[]{"shrinkAxisMask"})
                .build();



        map.put("begin",beginMapping);
        map.put("end",end);
        map.put("strides",strides);
        map.put("beginMask",beginMask);
        map.put("ellipsisMask",ellipsisMask);
        map.put("endMask",endMask);
        map.put("newAxisMask",newAxisMask);
        map.put("shrinkAxisMask",shrinkAxisMask);


        ret.put(tensorflowName(),map);

        return ret;
    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        return super.propertiesForFunction();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        throw new UnsupportedOperationException("Not yet supported/implemented");
    }

}
