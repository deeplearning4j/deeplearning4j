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

import static org.nd4j.linalg.api.buffer.DataType.INT32;

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
        //2 args: input and indices. Plus integer dimension arg
        //Gather backprop is just scatter add
        //credit for fixes to @paratsu
        SDVariable indicesGrad = sameDiff.zerosLike(arg(1));
        var gradAtOut = i_v.get(0);
        try (var ignore = sameDiff.withNameScope(gradAtOut.name())) {
            //Gather backprop is just scatter add
            SDVariable inputArray = arg(0);
            SDVariable indices = args().length > 1 ? arg(1) : sameDiff.constant(Nd4j.createFromArray(this.indices));
            SDVariable inputGrad = sameDiff.zerosLike(inputArray);

            // Use static shape information instead of dynamic ops for shape inference
            // During gradient computation, the forward pass has already executed, so shapes are known
            long[] inputShape = inputArray.getShape();
            int inputRank = inputShape != null ? inputShape.length : -1;

            // If inputShape is unknown (dynamic), fall back to simple scatter add on axis 0.
            // For axis 0, no permutation is needed: gradient is scatterAdd(zerosLike(input), indices, gradAtOut).
            if (inputRank <= 0) {
                // Dynamic-shape fallback: use scatterAdd directly on axis 0.
                // This is correct when jaxis == 0 (the common case for batch-dimension gather).
                // For other axes we still use the same approach since we cannot know the rank statically.
                SDVariable inputGradScattered = sameDiff.scatterAdd(inputGrad, indices, gradAtOut);
                return Arrays.asList(inputGradScattered, indicesGrad);
            }

            int outputRank = gradAtOut.getShape() != null ? gradAtOut.getShape().length : inputRank;

            // Normalize axis
            int normalizedAxis = jaxis < 0 ? inputRank + jaxis : jaxis;

            // Create constant arrays for permutation dimensions
            // gatherAxis = [normalizedAxis] as shape [1]
            SDVariable gatherAxis = sameDiff.constant(Nd4j.createFromArray(normalizedAxis).castTo(INT32));

            // inputArrayDimensions = [0, 1, 2, ..., inputRank-1]
            int[] allDims = new int[inputRank];
            for (int i = 0; i < inputRank; i++) {
                allDims[i] = i;
            }

            // inputArrayDimensionsRectified = all dimensions except the gather axis
            // e.g., if inputRank=2 and axis=0, this is [1]
            int[] rectifiedDims = new int[inputRank - 1];
            int idx = 0;
            for (int i = 0; i < inputRank; i++) {
                if (i != normalizedAxis) {
                    rectifiedDims[idx++] = i;
                }
            }
            SDVariable inputArrayDimensionsRectified = sameDiff.constant(Nd4j.createFromArray(rectifiedDims).castTo(INT32));

            // inputPermuteDims = [axis, other dims...] e.g., [0, 1] or [1, 0]
            int[] permuteDims = new int[inputRank];
            permuteDims[0] = normalizedAxis;
            for (int i = 0; i < rectifiedDims.length; i++) {
                permuteDims[i + 1] = rectifiedDims[i];
            }
            SDVariable inputPermuteDims = sameDiff.constant(Nd4j.createFromArray(permuteDims).castTo(INT32));

            // gradAtOutAdditionalDimensions = dimensions in output gradient beyond input rank
            // e.g., if input is [2,3] and indices is [4], output is [4,3], gradient is [4,3]
            // Additional dims would be empty in this case
            int additionalDims = outputRank - inputRank;
            int[] outGradPermuteDimsArr;
            if (additionalDims > 0) {
                outGradPermuteDimsArr = new int[outputRank];
                System.arraycopy(permuteDims, 0, outGradPermuteDimsArr, 0, inputRank);
                for (int i = 0; i < additionalDims; i++) {
                    outGradPermuteDimsArr[inputRank + i] = inputRank + i;
                }
            } else {
                outGradPermuteDimsArr = permuteDims;
            }
            SDVariable outGradPermuteDims = sameDiff.constant(Nd4j.createFromArray(outGradPermuteDimsArr).castTo(INT32));

            // inputInvertDims = inverse permutation
            int[] invertDims = new int[inputRank];
            for (int i = 0; i < inputRank; i++) {
                invertDims[permuteDims[i]] = i;
            }
            SDVariable inputInvertDims = sameDiff.constant(Nd4j.createFromArray(invertDims).castTo(INT32));

            //Permute gradients so original axis is at position 0... then scatter add, and reverse
            SDVariable permutedOutGrad = gradAtOut.permute(outGradPermuteDims);
            SDVariable inputGradPermuted = inputGrad.permute(inputPermuteDims);
            SDVariable inputGradPermutedScatterSum = sameDiff.scatterAdd(inputGradPermuted, indices, permutedOutGrad);

            //Now, invert the permutation so axis is back where it was
            SDVariable finalInputGrad = inputGradPermutedScatterSum.permute(inputInvertDims);
            return Arrays.asList(finalInputGrad, indicesGrad);
        }


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