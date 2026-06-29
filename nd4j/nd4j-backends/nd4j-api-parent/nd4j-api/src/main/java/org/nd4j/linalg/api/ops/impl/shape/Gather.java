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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;

/**
 * Gather op
 */
public class Gather extends DynamicCustomOp {

    protected int[] indices = null;
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
        this.jaxis = axis;
        this.indices = indices;
    }

    public Gather(SameDiff sameDiff, SDVariable input, SDVariable indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input, indices}, inPlace);
        addIArgument(axis);
        this.jaxis = axis;
    }

    public Gather(INDArray df, int[] indexes, int axis) {
        addInputArgument(df, Nd4j.createFromArray(indexes));
        addIArgument(axis);
        this.jaxis = axis;
        this.indices = indexes;
    }

    public Gather(INDArray df, INDArray indexes, int axis) {
        addInputArgument(df, indexes);
        addIArgument(axis);
        this.jaxis = axis;
        // FIXED: Removed incorrect line that referenced non-existent 'indices' variable
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
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        // ONNX Gather has an optional axis attribute, default is 0
        if (attributesForNode.containsKey("axis")) {
            long axis = attributesForNode.get("axis").getI();
            addIArgument(axis);
            this.jaxis = (int) axis;
        } else {
            // Default axis is 0
            addIArgument(0);
            this.jaxis = 0;
        }
    }

    @Override
    public void configureFromArguments() {
        if (!iArguments.isEmpty()) {
            this.jaxis = iArguments.get(0).intValue();
        } else {
            // Set default axis if no arguments provided
            this.jaxis = 0;
            addIArgument(0);
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
        // Gradient of gather w.r.t. the input:
        //   dInput[i] = sum of gradOut[j] for all j where indices[j] == i (along axis).
        // Duplicate indices accumulate — this is handled by GatherBp using atomicAdd on CUDA.
        SDVariable gradAtOut = i_v.get(0);
        SDVariable inputArg  = arg(0);
        // indices may be an SDVariable arg OR stored as int[] when constructed with int[] indices
        SDVariable indicesArg = args().length > 1
                ? arg(1)
                : sameDiff.constant(Nd4j.createFromArray(this.indices));
        SDVariable indicesGrad = sameDiff.zerosLike(indicesArg);

        // Obtain the forward input's shape as a runtime 1D int64 SDVariable.
        // SDVariable.shape() delegates to sameDiff.shape(this), which emits a Shape op
        // whose output values are the runtime shape dims — works for any shape, static or dynamic.
        SDVariable inputShapeVec = inputArg.shape();

        // Delegate the entire axis-aware scatter-add to the native gather_bp op.
        // inputShapeVec drives the value-dependent output shape in DECLARE_SHAPE_FN.
        SDVariable dInput = new GatherBp(sameDiff, inputShapeVec, indicesArg, gradAtOut, (long) jaxis)
                .outputVariable();

        return Arrays.asList(dInput, indicesGrad);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //Output type is same as (first) input type
        return Collections.singletonList(dataTypes.get(0));
    }

    /**
     * Gather output shape: replace the gather axis dimension with indices shape.
     * For data[D0, D1, ..., D_axis, ..., Dn] and indices[I0, I1, ..., Im],
     * output is [D0, D1, ..., I0, I1, ..., Im, ..., Dn]
     */
    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        if (oc == null || oc.numInputArguments() < 2) {
            return null;
        }

        INDArray data = oc.getInputArray(0);
        INDArray indices = oc.getInputArray(1);
        if (data == null || indices == null) {
            return null;
        }

        long[] dataShape = data.shape();
        long[] indicesShape = indices.shape();
        int dataRank = dataShape.length;

        // Get axis from iArgs or field
        List<Long> iArgs = oc.getIArguments();
        int axis;
        if (iArgs != null && !iArgs.isEmpty()) {
            axis = iArgs.get(0).intValue();
        } else {
            axis = this.jaxis;
        }

        // Normalize negative axis
        if (axis < 0) {
            axis += dataRank;
        }

        if (axis < 0 || axis >= dataRank) {
            return null; // Invalid axis - fall back to C++
        }

        // Output rank = data rank - 1 + indices rank
        // (we remove the axis dimension and insert indices shape)
        int outputRank = dataRank - 1 + indicesShape.length;
        long[] outputShape = new long[outputRank];

        // Copy dimensions before axis
        int outIdx = 0;
        for (int i = 0; i < axis; i++) {
            outputShape[outIdx++] = dataShape[i];
        }

        // Insert indices shape
        for (int i = 0; i < indicesShape.length; i++) {
            outputShape[outIdx++] = indicesShape[i];
        }

        // Copy dimensions after axis
        for (int i = axis + 1; i < dataRank; i++) {
            outputShape[outIdx++] = dataShape[i];
        }

        DataType dtype = data.dataType();
        long[] strides = Nd4j.getStrides(outputShape, 'c');
        boolean isEmpty = false;
        for (long dim : outputShape) {
            if (dim == 0) { isEmpty = true; break; }
        }
        LongShapeDescriptor descriptor = LongShapeDescriptor.fromShape(outputShape, strides, 1, 'c', dtype, isEmpty);
        DataBuffer shapeInfo = Shape.createShapeInformation(descriptor);
        return Collections.singletonList(shapeInfo);
    }
}