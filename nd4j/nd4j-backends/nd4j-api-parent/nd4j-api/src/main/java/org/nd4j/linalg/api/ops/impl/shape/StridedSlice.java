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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.bp.StridedSliceBp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.common.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

@Slf4j
public class StridedSlice extends DynamicCustomOp {
    private long[] begin;
    private long[] end;
    private long[] strides;
    private int beginMask;
    private int endMask;
    private int ellipsisMask;
    private int newAxisMask;
    private int shrinkAxisMask;

    public StridedSlice() {
    }

    public StridedSlice(SameDiff sameDiff, SDVariable in, int[] begin, int[] end, int[] strides){
        this(sameDiff, in, begin, end, strides, 0, 0, 0, 0, 0);
    }

    public StridedSlice(SameDiff sameDiff, SDVariable in, long[] begin, long[] end, long[] strides){
        this(sameDiff, in, begin, end, strides, 0, 0, 0, 0, 0);
    }

    public StridedSlice(SameDiff sameDiff, SDVariable in, @NonNull long[] begin, @NonNull long[] end, @NonNull long[] strides,
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

        //https://github.com/eclipse/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/parity_ops/strided_slice.cpp#L279
        addArguments();
    }

    public StridedSlice(SameDiff sameDiff, SDVariable in, @NonNull int[] begin, @NonNull int[] end, @NonNull int[] strides,
                        int beginMask, int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask){
        super(null, sameDiff, new SDVariable[]{in});
        this.begin = ArrayUtil.toLongArray(begin);
        this.end = ArrayUtil.toLongArray(end);
        this.strides = ArrayUtil.toLongArray(strides);
        this.beginMask = beginMask;
        this.endMask = endMask;
        this.ellipsisMask = ellipsisMask;
        this.newAxisMask = newAxisMask;
        this.shrinkAxisMask = shrinkAxisMask;
        addArguments();
        //https://github.com/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/parity_ops/strided_slice.cpp#L279

    }

    public StridedSlice(INDArray in, int[] begin, int[] end, int[] strides, int beginMask,
                        int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        this(in, ArrayUtil.toLongArray(begin), ArrayUtil.toLongArray(end), ArrayUtil.toLongArray(strides),
                beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    }

    public StridedSlice(INDArray in, long[] begin, long[] end, long[] strides, int beginMask,
                        int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        addInputArgument(in);
        this.begin = begin;
        this.end = end;
        this.strides = strides;
        this.beginMask = beginMask;
        this.endMask = endMask;
        this.ellipsisMask = ellipsisMask;
        this.newAxisMask = newAxisMask;
        this.shrinkAxisMask = shrinkAxisMask;
        addArguments();
    }


    public StridedSlice(SameDiff sd, SDVariable in, SDVariable begin, SDVariable end, SDVariable strides) {
        this(sd,in,begin,end,strides,0,0,0,0,0);
    }


    public StridedSlice(SameDiff sd, SDVariable in, SDVariable begin, SDVariable end, SDVariable strides,
                        int beginMask,
                        int endMask,
                        int ellipsisMask,
                        int newAxisMask,
                        int shrinkAxisMask) {
        super(sd,new SDVariable[]{in,begin,end,strides});
        this.beginMask = beginMask;
        this.endMask = endMask;
        this.ellipsisMask = ellipsisMask;
        this.newAxisMask = newAxisMask;
        this.shrinkAxisMask = shrinkAxisMask;
        addArguments();
    }

    public StridedSlice(INDArray in, INDArray begin, INDArray end, INDArray strides) {
        super(new INDArray[]{in,begin,end,strides},null);
        addArguments();
    }

    public StridedSlice(INDArray in, INDArray begin, INDArray end, INDArray strides, int beginMask, int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        super(new INDArray[]{in,begin,end,strides},null);
        this.beginMask = beginMask;
        this.endMask = endMask;
        this.ellipsisMask = ellipsisMask;
        this.newAxisMask = newAxisMask;
        this.shrinkAxisMask = shrinkAxisMask;
        addArguments();
    }


    private void addArguments() {
        //even without any specification java defaults to zero, we can safely call this as long as
        //the check is in place for begin, end and strides
        addIArgument(beginMask);
        addIArgument(ellipsisMask);
        addIArgument(endMask);
        addIArgument(newAxisMask);
        addIArgument(shrinkAxisMask);
        //these can  be inputs and maybe variables, it's not guaranteed that these will be specified
        if(begin != null)
            addIArgument(begin);
        if(end != null)
            addIArgument(end);
        if(strides != null)
            addIArgument(strides);
    }


    @Override
    public String opName() {
        return "strided_slice";
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

        // bit masks for this slice
        val bm = nodeDef.getAttrOrThrow("begin_mask");
        val xm = nodeDef.getAttrOrThrow("ellipsis_mask");
        val em = nodeDef.getAttrOrThrow("end_mask");
        val nm = nodeDef.getAttrOrThrow("new_axis_mask");
        val sm = nodeDef.getAttrOrThrow("shrink_axis_mask");

        beginMask = (int)bm.getI();
        ellipsisMask = (int) xm.getI();
        endMask = (int) em.getI();
        newAxisMask = (int) nm.getI();
        shrinkAxisMask = (int) sm.getI();

        addIArgument(beginMask);
        addIArgument(ellipsisMask);
        addIArgument(endMask);
        addIArgument(newAxisMask);
        addIArgument(shrinkAxisMask);
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
    public void configureFromArguments() {
        if(!iArguments.isEmpty()) {
            this.beginMask = iArguments.get(0).intValue();
            this.ellipsisMask = iArguments.get(1).intValue();
            this.endMask = iArguments.get(2).intValue();
            this.newAxisMask = iArguments.get(3).intValue();
            this.shrinkAxisMask = iArguments.get(4).intValue();

            int rankOfBeginEndStrides = (iArguments.size() - 5) / 3;
            begin = new long[rankOfBeginEndStrides];
            end = new long[rankOfBeginEndStrides];
            strides = new long[rankOfBeginEndStrides];
            for(int i = 0; i < rankOfBeginEndStrides; i++) {
                begin[i] = iArguments.get(i + 5);
                end[i] = iArguments.get(i + rankOfBeginEndStrides + 5);
                strides[i] = iArguments.get(i + (rankOfBeginEndStrides * 2) + 5);
            }

        }


    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("begin_mask")) {
            Long value = (Long) properties.get("begin_mask");
            this.beginMask = value.intValue();
        }

        if(properties.containsKey("ellipsis_mask")) {
            Long value = (Long) properties.get("ellipsis_mask");
            this.ellipsisMask = value.intValue();

        }

        if(properties.containsKey("end_mask")) {
            Long value = (Long) properties.get("end_mask");
            this.endMask = value.intValue();

        }

        if(properties.containsKey("shrink_axis_mask")) {
            Long value = (Long) properties.get("shrink_axis_mask");
            this.shrinkAxisMask = value.intValue();

        }

        if(properties.containsKey("new_axis_mask")) {
            Long value = (Long) properties.get("new_axis_mask");
            this.newAxisMask = value.intValue();
        }
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        if(args().length == 1) {
            //Array inputs for begin/end/strides
            return new StridedSliceBp(sameDiff, arg(), i_v.get(0), begin, end, strides, beginMask, endMask,
                    ellipsisMask, newAxisMask, shrinkAxisMask).outputs();
        } else {
            //SDVariable inputs for begin/end/strides
            return new StridedSliceBp(sameDiff, arg(), i_v.get(0), arg(1), arg(2), arg(3), beginMask, endMask,
                    ellipsisMask, newAxisMask, shrinkAxisMask).outputs();
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 4),
                "Expected 1 or 4 input datatypes for %s, got %s", getClass(), dataTypes);
        if(!dArguments.isEmpty()) {
            return Arrays.asList(dArguments.get(0));
        }
        //Output type is same as input type. 1 or 4 depending on whether using iargs or arrays (for TF import etc)
        return Collections.singletonList(dataTypes.get(0));
    }

}
