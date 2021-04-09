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

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.shade.guava.primitives.Ints;
import org.nd4j.shade.guava.primitives.Longs;
import lombok.NoArgsConstructor;
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.common.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

import static org.nd4j.common.util.ArrayUtil.*;

/**
 * TensorMmul
 * @author Adam Gibson
 */
@NoArgsConstructor
public class TensorMmul extends DynamicCustomOp {
    private int[][] axes;
    protected boolean addedEdges;
    protected MMulTranspose mMulTranspose;


    public TensorMmul(INDArray x, INDArray y, int[][] axes) {
        this(x,y,axes[0], axes[1], false, false, false);
    }

    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     */
    public TensorMmul(INDArray x, INDArray y, INDArray z, int[][] axes) {
        this(x, y, axes[0], axes[1], false, false, false);
    }

    public TensorMmul(INDArray x, INDArray y, int[] dimensionsX, int[] dimensionsY,
                      boolean transposeX, boolean transposeY, boolean transposeZ) {
        super(null,new INDArray[]{x, y},null);
        this.axes = new int[][]{dimensionsX, dimensionsY};
        addIArgument(dimensionsX.length);
        addIArgument(dimensionsX);
        addIArgument(dimensionsY.length);
        addIArgument(dimensionsY);
        addBArgument(transposeX, transposeY, transposeZ);
    }

    public TensorMmul(SameDiff sameDiff,
                      SDVariable i_v1,
                      SDVariable i_v2,
                      int[][] dimensions) {
        this(sameDiff,i_v1,i_v2,dimensions,MMulTranspose.allFalse());
    }

    public TensorMmul(SameDiff sameDiff,
                      SDVariable i_v1,
                      SDVariable i_v2,
                      int[][] dimensions,
                      MMulTranspose mMulTranspose) {
        super(null, sameDiff, new SDVariable[]{i_v1,i_v2});
        this.sameDiff = sameDiff;
        this.mMulTranspose = mMulTranspose;
        this.axes = dimensions;
        if(!addedEdges && sameDiff.getOutputsForOp(this) == null) {
            addedEdges = true;
        }

        addIArgument(dimensions[0].length);
        addIArgument(dimensions[0]);
        addIArgument(dimensions[1].length);
        addIArgument(dimensions[1]);
    }

    public TensorMmul(SameDiff sameDiff, SDVariable x, SDVariable y, int[] dimensionsX,
                      int[] dimensionsY, boolean transposeX, boolean transposeY, boolean transposeZ) {
        super(null, sameDiff, new SDVariable[]{x,y});
        this.sameDiff = sameDiff;
        this.axes = new int[][]{dimensionsX, dimensionsY};
        addIArgument(dimensionsX.length);
        addIArgument(dimensionsX[0]);
        addIArgument(dimensionsY.length);
        addIArgument(dimensionsY[0]);
        addBArgument(transposeX, transposeY, transposeZ);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        return Arrays.asList(new TensorMmulBp(sameDiff, larg(), rarg(), gradients.get(0), axes ).outputVariables());
    }

    @Override
    public String opName() {
        return "tensordot";
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
        /**
         * name: "MatMul"
         op: "MatMul"
         input: "input"
         input: "Variable/read"
         attr {
         key: "transpose_b"
         value {
         b: false
         }
         }
         attr {
         key: "transpose_a"
         value {
         b: false
         }
         }
         attr {
         key: "T"
         value {
         type: DT_FLOAT
         }
         }

         */

        val isTransposeA = attributesForNode.get("transpose_a").getB();
        val isTransposeB = attributesForNode.get("transpose_b").getB();
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeA(isTransposeA).transposeB(isTransposeB)
                .build();
        this.mMulTranspose = mMulTranspose;
        val args = args();
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        val isTransposeA = !attributesForNode.containsKey("transA") ? false : attributesForNode.get("transA").getI() > 0;
        val isTransposeB = !attributesForNode.containsKey("transB") ? false : attributesForNode.get("transB").getI() > 0;
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeA(isTransposeA).transposeB(isTransposeB)
                .build();
        this.mMulTranspose = mMulTranspose;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TensorMmul that = (TensorMmul) o;

        if (addedEdges != that.addedEdges) return false;
        if (!Arrays.deepEquals(axes, that.axes)) return false;
        return mMulTranspose != null ? mMulTranspose.equals(that.mMulTranspose) : that.mMulTranspose == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + Arrays.deepHashCode(axes);
        result = 31 * result + (addedEdges ? 1 : 0);
        result = 31 * result + (mMulTranspose != null ? mMulTranspose.hashCode() : 0);
        return result;
    }


    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public String onnxName() {
        return "Gemm";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2, "Expected exactly 2 input data types for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
