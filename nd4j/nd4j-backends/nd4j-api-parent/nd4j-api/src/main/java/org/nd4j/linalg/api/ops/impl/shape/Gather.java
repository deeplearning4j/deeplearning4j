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

import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Gather op
 */
public class Gather extends DynamicCustomOp {

    protected int[] indices;
    protected int jaxis = 0;

    public Gather() {
    }

    public Gather(SameDiff sameDiff, SDVariable df, SDVariable indices, int axis) {
        this(sameDiff, df, indices, axis, false);
    }

    public Gather(SameDiff sameDiff, SDVariable df, int[] indices, int axis) {
        this(sameDiff, df, indices, axis, false);
    }

    public Gather(SameDiff sameDiff, SDVariable input, int[] indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input, sameDiff.constant(Nd4j.createFromArray(indices))}, inPlace);

        addIArgument(axis);
        addIArgument(indices);
        this.jaxis = axis;
        this.indices = indices;
    }

    public Gather(SameDiff sameDiff, SDVariable input, SDVariable indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input, indices}, inPlace);
        addIArgument(axis);
        this.jaxis = axis;
    }

    public Gather(INDArray df, int[] indexes, int axis) {
        addInputArgument(df);
        addIArgument(axis);
        addIArgument(indexes);
        this.jaxis = axis;
        this.indices = indices;
    }

    public Gather(INDArray df, INDArray indexes, int axis) {
        addInputArgument(df, indexes);
        addIArgument(axis);
        this.jaxis = axis;
        this.indices = indices;
    }

    @Override
    public String onnxName() {
        return "Gather";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]{"Gather", "GatherV2"};
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }

    @Override
    public void configureFromArguments() {
        if(!iArguments.isEmpty()) {
            this.jaxis = iArguments.get(0).intValue();
        }
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        val broadcast = PropertyMapping.builder()
                .onnxAttrName("indices")
                .tfInputPosition(1)
                .propertyNames(new String[]{"indices"}).build();

        map.put("indices", broadcast);

        ret.put(tensorflowNames()[0], map);
        ret.put(onnxName(), map);

        Map<String, PropertyMapping> map2 = new HashMap<>();
        val broadcast2 = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"indices"}).build();
        map2.put("indices", broadcast2);

        val axis2 = PropertyMapping.builder()
                .tfInputPosition(2)
                .propertyNames(new String[]{"axis"}).build();
        map2.put("axis", axis2);

        ret.put("GatherV2", map2);


        return ret;
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("dimensions")) {
            Long dimensions = (Long) properties.get("dimensions");
            this.jaxis = dimensions.intValue();
        }
    }

    @Override
    public String opName() {
        return "gather";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //2 args: input and indices. Plus integer dimension arg
        //Gather backprop is just scatter add
        SDVariable indicesSize = sameDiff.expandDims(args()[1].length(),0);
        SDVariable paramsShape = sameDiff.shape(args()[0]);
        paramsShape = paramsShape.reshape(paramsShape.length());
        SDVariable indicesGrad = sameDiff.zerosLike(arg(1));

        if(jaxis == 0) {
            SDVariable paramsTailShape = paramsShape.getView(SDIndex.interval(sameDiff.constant(1)
                    , sameDiff.constant(1),paramsShape.length()));
            SDVariable valueShape = sameDiff.concat(0,indicesSize,paramsTailShape);
            SDVariable values = sameDiff.reshape(i_v.get(0),valueShape);
            SDVariable indices = sameDiff.flatten(args()[1]);
            SDVariable retGrad = sameDiff.zerosLike(arg());
            SDVariable put = retGrad.put(indices,values,indices).reshape(arg().shape());
            /**
             * TODO: figure out a better way to do a mass assign.
             * We can't match the speed of a sparse gradient so we need to figure out the best way to
             * achieve this with a dense representation.
             *
             * This would ideally be similar to nd4j's put(indices)
             */
            return Arrays.asList(put, indicesGrad);
        } else {
            SDVariable batchDims = sameDiff.constant(0);
            SDVariable outerShape = paramsShape.getView(SDIndex.interval(0,jaxis));
            SDVariable innerShape = paramsShape.getView(
                    SDIndex.interval(sameDiff.constant(jaxis),paramsShape.length()),SDIndex.interval(sameDiff.constant(1),sameDiff.constant(-1)));
            SDVariable valueShape = sameDiff.concat(0,outerShape,
                    sameDiff.constant(-1).castTo(outerShape.dataType()),
                    innerShape.castTo(outerShape.dataType()));


            /**
             * Blow grad up to match values shape, values shape is  not wrong
             */
            SDVariable valuesDims = valueShape.length();
            SDVariable axisDims = outerShape.length();

            SDVariable outerBatchIndices = sameDiff.range(0,0,0,DataType.INT64);
            SDVariable batchAxisIndices = sameDiff.range(batchDims,axisDims, sameDiff.constant(1),DataType.INT64);
            SDVariable innerAxisIndices = sameDiff.range(axisDims.add(1.0),valuesDims,sameDiff.constant(1),DataType.INT64);

            SDVariable indices = sameDiff.reshape(args()[1],indicesSize);

            SDVariable put = sameDiff.unsortedSegmentSum(i_v.get(0), sameDiff.range(sameDiff.constant(0),sameDiff.sizeAt(i_v.get(0),0),sameDiff.constant(1),DataType.INT64), sameDiff.sizeAt(i_v.get(0),0));
            SDVariable values = sameDiff.reshape(put,valueShape);



            SDVariable transposeDims = sameDiff.concat("transposeConcat",0,outerBatchIndices,axisDims,batchAxisIndices,innerAxisIndices);
            SDVariable valuesTranspose = sameDiff.permute(values,transposeDims);

            /**
             * Batch gather grad
             */

            SDVariable paramsGrad = sameDiff.unsortedSegmentSum(valuesTranspose,indices,paramsShape.get(SDIndex.point(jaxis)));
            SDVariable invertTransposeDims = sameDiff.concat(0,outerBatchIndices.castTo(DataType.INT64),batchAxisIndices.add(1).castTo(DataType.INT64),batchDims.castTo(DataType.INT64),innerAxisIndices.castTo(DataType.INT64));
            paramsGrad = sameDiff.permute(paramsGrad,invertTransposeDims);


            return Arrays.asList(paramsGrad, indicesGrad);
        }

    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //Output type is same as (first) input type
        return Collections.singletonList(dataTypes.get(0));
    }
}
