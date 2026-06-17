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

package org.nd4j.linalg.api.ops.impl.reduce;

import lombok.EqualsAndHashCode;
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.util.*;

@EqualsAndHashCode
public class Mmul extends DynamicCustomOp {

    protected MMulTranspose mt;
    protected double alpha = 1.0;
    protected double beta = 0.0;

    /**
     *
     * @param sameDiff
     * @param i_v1
     * @param i_v2
     * @param mt
     */
    public Mmul(SameDiff sameDiff,
                SDVariable i_v1,
                SDVariable i_v2,
                MMulTranspose mt) {
        super(null,sameDiff,new SDVariable[]{i_v1,i_v2});
        this.mt = mt;
        addIArgument(ArrayUtil.fromBoolean(mt.isTransposeA()), ArrayUtil.fromBoolean(mt.isTransposeB()), ArrayUtil.fromBoolean(mt.isTransposeResult()));
        addTArgument(alpha, beta);
    }


    /**
     *
     * @param sameDiff
     * @param i_v1
     * @param i_v2
     */
    public Mmul(SameDiff sameDiff,
                SDVariable i_v1,
                SDVariable i_v2) {
        this(sameDiff,i_v1,i_v2,MMulTranspose.allFalse());
    }

    public Mmul(INDArray x,
                INDArray y,
                INDArray z,
                double alpha,
                double beta,
                MMulTranspose mt) {
        addInputArgument(x, y);

        if (z != null)
            addOutputArgument(z);

        if (mt != null) {
            this.mt = mt;
            addIArgument(ArrayUtil.fromBoolean(mt.isTransposeA()),
                    ArrayUtil.fromBoolean(mt.isTransposeB()),
                    ArrayUtil.fromBoolean(mt.isTransposeResult()));
        }

        this.alpha = alpha;
        this.beta = beta;

        addTArgument(alpha, beta);
    }

    /**
     *
     * @param x
     * @param y
     * @param z
     */
    public Mmul(INDArray x,
                INDArray y,
                INDArray z,
                MMulTranspose mt) {
        this(x, y, z, 1.0, 0.0, mt);
    }

    public Mmul(INDArray x, INDArray y, boolean transposeX, boolean transposeY,  boolean transposeZ) {
        this(x, y, 1.0, 0.0, transposeX, transposeY, transposeZ);
    }

    public Mmul(INDArray x, INDArray y, double alpha, double beta, boolean transposeX, boolean transposeY,  boolean transposeZ) {
        addInputArgument(x, y);
        addIArgument(ArrayUtil.fromBoolean(transposeX),
                ArrayUtil.fromBoolean(transposeY),
                ArrayUtil.fromBoolean(transposeZ));
        mt = MMulTranspose.builder().transposeA(transposeX).transposeB(transposeY).transposeResult(transposeZ).build();
        addTArgument(alpha, beta);
        this.alpha = alpha;
        this.beta = beta;
    }

    public Mmul(INDArray x, INDArray y, double alpha, double beta) {
        this(x,y,null, alpha, beta,null);
    }

    public Mmul(INDArray x, INDArray y) {
        this(x, y, 1.0, 0.0);
    }

    public Mmul(SameDiff sameDiff, SDVariable x, SDVariable y, boolean transposeX, boolean transposeY,
                boolean transposeZ) {
        super(null,sameDiff,new SDVariable[]{x,y});
        addIArgument(ArrayUtil.fromBoolean(transposeX),
                ArrayUtil.fromBoolean(transposeY),
                ArrayUtil.fromBoolean(transposeZ));

        addTArgument(alpha, beta);
        mt = MMulTranspose.builder().transposeA(transposeX).transposeB(transposeY).transposeResult(transposeZ).build();
    }

    public Mmul() {
    }

    public Mmul(SameDiff sd, SDVariable a, SDVariable b, double alpha, double beta, boolean transA, boolean transB) {
        super(null,sd,new SDVariable[]{a,b});
        addIArgument(ArrayUtil.fromBoolean(transA),
                ArrayUtil.fromBoolean(transB),
                ArrayUtil.fromBoolean(false));
        this.alpha = alpha;
        this.beta = beta;
        addTArgument(alpha, beta);
        mt = MMulTranspose.builder().transposeA(transA).transposeB(transB).transposeResult(false).build();

    }

    public Mmul(INDArray a, INDArray b, double alpha, double beta, boolean transA, boolean transB) {
        addInputArgument(a, b);
        addIArgument(ArrayUtil.fromBoolean(transA),
                ArrayUtil.fromBoolean(transB),
                ArrayUtil.fromBoolean(false));
        this.alpha = alpha;
        this.beta = beta;
        addTArgument(alpha, beta);
        mt = MMulTranspose.builder().transposeA(transA).transposeB(transB).transposeResult(false).build();

    }

    @Override
    public Object getValue(Field property) {
        if (mt == null) {
            mt = MMulTranspose.builder().build();
        }

        return mt.getValue(property);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        if(mt == null)
            return Collections.emptyMap();
        return mt.toProperties();
    }

    @Override
    public void configureFromArguments() {
        this.mt = MMulTranspose.builder()
                .transposeA(numIArguments() > 0 && getIArgument(0) > 0)
                .transposeB(numIArguments() > 1 && getIArgument(1) > 0)
                .transposeResult(numIArguments() > 2 && getIArgument(2) > 0)
                .build();
    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "mt";
    }

    public void setPropertiesForFunction(Map<String,Object> properties) {
        if(mt == null)
            mt = MMulTranspose.builder().build();
        mt.setProperties(properties);
    }

    /**
     * For a 2D matrix of shape (M, N) we return (N, M).
     * For a 3D matrix with leading mini-batch dimension (mb, M, N)
     * we return (mb, N, M)
     *
     * @param shape input shape array
     * @return
     */
    public long[] transposeShapeArray(long[] shape) {
        if (shape.length == 2) {
            return ArrayUtil.reverseCopy(shape);
        } else if (shape.length == 3) {
            return new long[] {shape[0], shape[2], shape[1]};
        } else {
            throw new IllegalArgumentException("Matrix input has to be of length 2 or 3, got: " + shape.length );
        }

    }

    @Override
    public String onnxName() {
        return "MatMul";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"MatMul", "BatchMatMul", "BatchMatMulV2"};
    }



    @Override
    public String opName() {
        return "matmul";
    }



    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);

        boolean isTransposeA;
        boolean isTransposeB;
        if(nodeDef.getOp().equalsIgnoreCase("MatMul")){
            isTransposeA = attributesForNode.get("transpose_a").getB();
            isTransposeB = attributesForNode.get("transpose_b").getB();

        } else {
            //BatchMatMul, BatchMatMulV2
            //In practice, BatchMatMul seems to use "adj_x" and "adj_y" instead of "transpose_a" and "transpose_b"
            if(attributesForNode.containsKey("transpose_a")){
                isTransposeA = attributesForNode.get("transpose_a").getB();
            } else {
                isTransposeA = attributesForNode.get("adj_x").getB();
            }
            if(attributesForNode.containsKey("transpose_b")){
                isTransposeB = attributesForNode.get("transpose_b").getB();
            } else {
                isTransposeB = attributesForNode.get("adj_y").getB();
            }
        }
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeA(isTransposeA).transposeB(isTransposeB)
                .build();
        this.mt = mMulTranspose;
        iArguments.clear();
        addIArgument(ArrayUtil.fromBoolean(mt.isTransposeA()), ArrayUtil.fromBoolean(mt.isTransposeB()));
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        val isTransposeA = !attributesForNode.containsKey("transA") ? false : attributesForNode.get("transA").getI() > 0;
        val isTransposeB = !attributesForNode.containsKey("transB") ? false : attributesForNode.get("transB").getI() > 0;
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeA(isTransposeA).transposeB(isTransposeB)
                .build();
        this.mt = mMulTranspose;
    }





    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        return Arrays.asList(new MmulBp(sameDiff, arg(0), arg(1), gradients.get(0), mt).outputVariables());
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val transposeA = PropertyMapping.builder()
                .onnxAttrName("transA")
                .tfAttrName("transpose_a")
                .propertyNames(new String[]{"transposeA"})
                .build();

        val transposeB = PropertyMapping.builder()
                .onnxAttrName("transB")
                .tfAttrName("transpose_b")
                .propertyNames(new String[]{"transposeB"})
                .build();

        map.put("transposeA",transposeA);
        map.put("transposeB",transposeB);

        for(String s : tensorflowNames()) {
            ret.put(s,map);
        }
        ret.put(onnxName(),map);

        return ret;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        if(!dArguments.isEmpty())
            return Collections.singletonList(dArguments.get(0));
        Preconditions.checkState(dataTypes != null && dataTypes.size() >= 2, "Expected at least 2 inputs to mmul op, got %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType() && dataTypes.get(1).isFPType(), "Inputs to mmul op must both be a floating" +
                "point type: got %s", dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }

    /**
     * Calculate matmul output shape directly from input arrays in Java, avoiding JNI overhead.
     * For A[..., M, K] x B[..., K, N] -> output[..., M, N] with batch dimension broadcasting.
     */
    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        if (oc == null || oc.numInputArguments() < 2) {
            return null;
        }

        INDArray a = oc.getInputArray(0);
        INDArray b = oc.getInputArray(1);
        if (a == null || b == null) {
            return null;
        }

        long[] aShape = a.shape();
        long[] bShape = b.shape();

        if (aShape.length < 2 || bShape.length < 2) {
            return null; // Fall back to C++ for edge cases
        }

        // Get transpose flags from iArgs
        List<Long> iArgs = oc.getIArguments();
        boolean transposeA = iArgs != null && iArgs.size() > 0 && iArgs.get(0) != 0;
        boolean transposeB = iArgs != null && iArgs.size() > 1 && iArgs.get(1) != 0;
        boolean transposeResult = iArgs != null && iArgs.size() > 2 && iArgs.get(2) != 0;

        int aRank = aShape.length;
        int bRank = bShape.length;

        // Get M, K from A (after transpose if needed)
        long aM = transposeA ? aShape[aRank - 1] : aShape[aRank - 2];
        long aK = transposeA ? aShape[aRank - 2] : aShape[aRank - 1];

        // Get K, N from B (after transpose if needed)
        long bK = transposeB ? bShape[bRank - 1] : bShape[bRank - 2];
        long bN = transposeB ? bShape[bRank - 2] : bShape[bRank - 1];

        // K dimensions must match
        if (aK != bK) {
            return null; // Dimensions don't match - fall back to C++ for error handling
        }

        // Compute batch dimensions with broadcasting
        int maxBatchRank = Math.max(aRank, bRank) - 2;
        long[] outputShape = new long[maxBatchRank + 2];

        // Broadcast batch dimensions (from the end, excluding last 2)
        for (int i = 0; i < maxBatchRank; i++) {
            int aIdx = aRank - 3 - i;
            int bIdx = bRank - 3 - i;
            long aDim = aIdx >= 0 ? aShape[aIdx] : 1;
            long bDim = bIdx >= 0 ? bShape[bIdx] : 1;

            if (aDim != bDim && aDim != 1 && bDim != 1) {
                return null; // Not broadcastable - fall back to C++
            }
            outputShape[maxBatchRank - 1 - i] = Math.max(aDim, bDim);
        }

        // Set M and N dimensions
        outputShape[maxBatchRank] = aM;
        outputShape[maxBatchRank + 1] = bN;

        // Apply result transpose if needed
        if (transposeResult) {
            long temp = outputShape[maxBatchRank];
            outputShape[maxBatchRank] = outputShape[maxBatchRank + 1];
            outputShape[maxBatchRank + 1] = temp;
        }

        DataType dtype = a.dataType();
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
