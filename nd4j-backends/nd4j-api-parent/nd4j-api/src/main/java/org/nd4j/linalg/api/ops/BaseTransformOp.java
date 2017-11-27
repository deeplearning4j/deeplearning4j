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

package org.nd4j.linalg.api.ops;

import lombok.extern.slf4j.Slf4j;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A base op for basic getters and setters
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseTransformOp extends BaseOp implements TransformOp {


    public BaseTransformOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2) {
        this(sameDiff,i_v1,i_v2,false);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           boolean inPlace) {
        super(sameDiff,inPlace,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);
            f().validateFunctionReference(i_v1);
            f().validateFunctionReference(i_v2);
            this.sameDiff = sameDiff;
            this.inPlace = inPlace;
            addAsNewVertexId();
            sameDiff.putShapeForVertexId(vertexId,i_v1.getResultShape());
            sameDiff.associateFunctionsAsArgs(new DifferentialFunction[] {i_v1,i_v2},this);
            f().addFunctionEdges(this);

        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }


    }

    public BaseTransformOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public BaseTransformOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           Object[] extraArgs) {
        super(sameDiff,extraArgs);
        if (i_v1 != null && i_v2 != null) {

            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);
            this.sameDiff = sameDiff;
            addAsNewVertexId();
            sameDiff.putShapeForVertexId(vertexId,i_v1.getResultShape());
            sameDiff.associateFunctionsAsArgs(new DifferentialFunction[] {i_v1,i_v2},this);

            f().addFunctionEdges(this);

        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }

    }




    public BaseTransformOp(SameDiff sameDiff,DifferentialFunction i_v,boolean inPlace) {
        this(sameDiff,i_v,i_v.getResultShape(),inPlace,null);
    }

    public BaseTransformOp(SameDiff sameDiff,
                           DifferentialFunction i_v,
                           int[] shape,
                           boolean inPlace,
                           Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        ;
        if (i_v != null) {
            f().validateFunctionReference(i_v);
            f().validateDifferentialFunctionsameDiff(i_v);
            addAsNewVertexId();
            sameDiff.putShapeForVertexId(vertexId,shape);
            sameDiff.associateFunctionsAsArgs(new DifferentialFunction[] {sameDiff.setupFunction(i_v)},this);
            f().addFunctionEdges(this);
        } else {
            throw new IllegalArgumentException("Input must not null variable.");
        }

    }


    public BaseTransformOp(SameDiff sameDiff,
                           DifferentialFunction i_v,
                           Object[] extraArgs) {
        this(sameDiff,i_v,i_v.getResultShape(),false,extraArgs);
    }



    public BaseTransformOp(INDArray x, INDArray z) {
        super(x, z);
        LinAlgExceptions.assertSameLength(x, z);
        LinAlgExceptions.assertSameShape(x, z);
    }

    public BaseTransformOp() {}

    public BaseTransformOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public BaseTransformOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        if (y != null)
            LinAlgExceptions.assertSameLength(x, y);
        LinAlgExceptions.assertSameLength(x, z);

    }

    public BaseTransformOp(INDArray x) {
        super(x);
    }

    @Override
    public Type opType() {
        if(args() == null || args().length == 1)
            return Type.TRANSFORM;
        else if(args().length == 2)
            return Type.PAIRWISE;

        else throw new ND4JIllegalStateException("Illegal number of args (can only be 1 or 2)");
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        ret.add(arg().getResultShape());
        return ret;
    }

    @Override
    protected void addAsNewVertexId() {
        super.addAsNewVertexId();
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);


    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);

    }

}
