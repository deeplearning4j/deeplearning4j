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
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.primitives.Longs;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Transpose extends DynamicCustomOp {
    protected long[] permuteDims;

    public Transpose(SameDiff sameDiff, SDVariable i_v) {
        super(null, sameDiff, new SDVariable[]{i_v});
    }

    public Transpose(SameDiff sameDiff, SDVariable in, long[] permuteDims){
        super(null, sameDiff, new SDVariable[]{in});
        this.permuteDims = permuteDims;
    }

    protected Transpose(SameDiff sameDiff, SDVariable in, SDVariable permuteDims){
        super(null, sameDiff, new SDVariable[]{in, permuteDims});
    }

    public Transpose(INDArray input, INDArray result){
        super(null, new INDArray[]{input}, result == null ? null : new INDArray[]{result}, null, (List<Long>) null);
    }

    public Transpose(INDArray input){
        this(input, null);
    }

    public Transpose() {
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new LinkedHashMap<>();
        Map<String, PropertyMapping> map = new LinkedHashMap<>();

        val mapping = PropertyMapping.builder()
                .onnxAttrName("perm")
                .propertyNames(new String[]{"permuteDims"})
                .tfInputPosition(1)
                .build();


        map.put("permuteDims", mapping);
        ret.put(tensorflowName(), map);
        ret.put(onnxName(), map);
        return ret;
    }


    @Override
    public String opName() {
        return "transpose";
    }

    @Override
    public String onnxName() {
        return "Transpose";
    }

    @Override
    public String tensorflowName() {
        return "Transpose";
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        if (!attributesForNode.containsKey("perm")) {

        } else
            this.permuteDims = Longs.toArray(attributesForNode.get("perm").getIntsList());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable ret;
        if(permuteDims == null) {
            ret = sameDiff.transpose(i_v.get(0));
        } else {
            long[] reverse = ArrayUtil.invertPermutation(permuteDims);
            ret = sameDiff.permute(i_v.get(0), reverse);
        }
        return Collections.singletonList(ret);
    }

    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected list with 1 or 2 datatype for %s, got %s", getClass(), dataTypes);
        if(dArguments != null && !dArguments.isEmpty())
            return Collections.singletonList(dArguments.get(0));
        //Output type is same as input type. Second input is permute dimensions as array
        return Collections.singletonList(dataTypes.get(0));
    }

    /**
     * Transpose/Permute output shape is input dimensions reordered according to permuteDims.
     * For simple transpose (no permuteDims), dimensions are reversed.
     */
    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        if (oc == null || oc.numInputArguments() < 1) {
            return null;
        }

        // If there's a second input (dynamic permutation), fall back to C++
        // because we'd need to sync and read the values from the device
        if (oc.numInputArguments() > 1) {
            return null;
        }

        INDArray input = oc.getInputArray(0);
        if (input == null) {
            return null;
        }

        long[] inputShape = input.shape();
        int rank = inputShape.length;
        long[] outputShape = new long[rank];

        // Get permutation from iArgs or permuteDims field
        List<Long> iArgs = oc.getIArguments();
        long[] perm = null;
        if (iArgs != null && !iArgs.isEmpty()) {
            perm = new long[iArgs.size()];
            for (int i = 0; i < iArgs.size(); i++) {
                perm[i] = iArgs.get(i);
            }
        } else if (permuteDims != null && permuteDims.length > 0) {
            perm = permuteDims;
        }

        if (perm != null && perm.length != rank) {
            // Rank mismatch — fall back to C++ which has adaptation logic
            return null;
        }

        if (perm != null) {
            // Apply permutation
            for (int i = 0; i < rank; i++) {
                int srcAxis = (int) perm[i];
                if (srcAxis < 0) srcAxis += rank;
                if (srcAxis < 0 || srcAxis >= rank) {
                    return null; // Invalid permutation - fall back to C++
                }
                outputShape[i] = inputShape[srcAxis];
            }
        } else {
            // Default transpose: reverse dimensions
            for (int i = 0; i < rank; i++) {
                outputShape[i] = inputShape[rank - 1 - i];
            }
        }

        DataType dtype = input.dataType();
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
