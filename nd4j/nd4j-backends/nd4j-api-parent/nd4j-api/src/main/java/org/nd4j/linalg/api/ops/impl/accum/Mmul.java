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

package org.nd4j.linalg.api.ops.impl.accum;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Matrix multiplication/dot product
 *
 * @author Adam Gibson
 */
public class Mmul extends DynamicCustomOp {

    protected MMulTranspose mMulTranspose;

    /**
     *
     * @param sameDiff
     * @param i_v1
     * @param i_v2
     * @param mMulTranspose
     */
    public Mmul(SameDiff sameDiff,
                SDVariable i_v1,
                SDVariable i_v2,
                MMulTranspose mMulTranspose) {
        super(null,sameDiff,new SDVariable[]{i_v1,i_v2});
        this.mMulTranspose = mMulTranspose;
        addIArgument(ArrayUtil.fromBoolean(mMulTranspose.isTransposeA()), ArrayUtil.fromBoolean(mMulTranspose.isTransposeB()));
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

    /**
     *
     * @param x
     * @param y
     * @param z
     */
    public Mmul(INDArray x,
                INDArray y,
                INDArray z,
                MMulTranspose mMulTranspose) {
        super(null, new INDArray[]{x, y}, z == null ? null : new INDArray[]{z});
        if (mMulTranspose != null) {
          this.mMulTranspose = mMulTranspose;
          addIArgument(ArrayUtil.fromBoolean(mMulTranspose.isTransposeA()),
                       ArrayUtil.fromBoolean(mMulTranspose.isTransposeB()));
        }
    }


    public Mmul() {}


    @Override
    public List<long[]> calculateOutputShape() {
        if(mMulTranspose == null)
            mMulTranspose = MMulTranspose.allFalse();
        List<long[]> ret = new ArrayList<>(1);
        long[] aShape = mMulTranspose.isTransposeA() ? ArrayUtil.reverseCopy(larg().getShape()) : larg().getShape();
        long[] bShape = mMulTranspose.isTransposeB() ? ArrayUtil.reverseCopy(rarg().getShape()) : rarg().getShape();
        if(Shape.isPlaceholderShape(aShape) || Shape.isPlaceholderShape(bShape))
            return Collections.emptyList();

        if(aShape != null && bShape != null) {
            val shape =  Shape.getMatrixMultiplyShape(aShape,bShape);
            ret.add(shape);
        }
        if(!ret.isEmpty()) {
            for(int i = 0; i < ret.get(0).length; i++) {
                if(ret.get(0)[i] < 1)
                    throw new ND4JIllegalStateException("Invalid shape computed at index " +  i);
            }
        }
        return ret;
    }


    @Override
    public String onnxName() {
        return "MatMul";
    }

    @Override
    public String tensorflowName() {
        return "MatMul";
    }



    @Override
    public String opName() {
        return "mmul";
    }



    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
        val isTransposeA = attributesForNode.get("transpose_a").getB();
        val isTransposeB = attributesForNode.get("transpose_b").getB();
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeA(isTransposeA).transposeB(isTransposeB)
                .build();
        this.mMulTranspose = mMulTranspose;
        val args = args();
        for(val arg : args) {
            if(sameDiff.isPlaceHolder(arg.getVarName()) || arg.getShape() == null) {
                sameDiff.addPropertyToResolve(this,arg.getVarName());
            }
        }
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val isTransposeA = !attributesForNode.containsKey("transA") ? false : attributesForNode.get("transA").getI() > 0;
        val isTransposeB = !attributesForNode.containsKey("transB") ? false : attributesForNode.get("transB").getI() > 0;
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeA(isTransposeA).transposeB(isTransposeB)
                .build();
        this.mMulTranspose = mMulTranspose;
    }





    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        List<SDVariable> ret = new ArrayList<>();
        SDVariable setup = sameDiff.setupFunction(i_v1.get(0));
        SDVariable gradWrtX = sameDiff.setupFunction(f().reshape(f().mmul(setup,rarg(),
                MMulTranspose.builder()
                        .transposeB(!mMulTranspose.isTransposeB())
                        .transposeResult(mMulTranspose.isTransposeA())
                        .build()),larg().getShape()));

        SDVariable gradWrtY = sameDiff.setupFunction(f().reshape(f().mmul(larg(),setup,
                MMulTranspose.builder()
                        .transposeA(!mMulTranspose.isTransposeA())
                        .transposeResult(mMulTranspose.isTransposeB())
                        .build()),rarg().getShape()));

        ret.add(gradWrtX);
        ret.add(gradWrtY);
        return ret;
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

        ret.put(tensorflowName(),map);
        ret.put(onnxName(),map);

        return ret;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Mmul mmul = (Mmul) o;

        return mMulTranspose != null ? mMulTranspose.equals(mmul.mMulTranspose) : mmul.mMulTranspose == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (mMulTranspose != null ? mMulTranspose.hashCode() : 0);
        return result;
    }
}

