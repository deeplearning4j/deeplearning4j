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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

/**
 * Base class for accumulation, initiates the initial entry
 * with respect to the child class. Also contains baseline fields
 * for the over all field with accumulation.
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseAccumulation extends BaseOp implements Accumulation {
    protected Number finalResult;
    protected IComplexNumber finalResultComplex;
    protected boolean applyFinalTransform = true;
    protected boolean isComplex = false;

    public BaseAccumulation(SameDiff sameDiff,
                            DifferentialFunction i_v,
                            int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            this.args = new DifferentialFunction[] {i_v};
            this.dimensions = dimensions;
            this.shape = Shape.getReducedShape(i_v.getResultShape(),dimensions);
            f().validateDifferentialFunctionsameDiff(i_v);
            addAsNewVertexId();
            f().addFunctionEdges(this);
            this.opState.setAxes(dimensions);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

    }

    public BaseAccumulation(SameDiff sameDiff,
                            DifferentialFunction i_v,
                            DifferentialFunction i_v2,
                            int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            this.args = new DifferentialFunction[] {i_v,i_v2};
            this.dimensions = dimensions;
            this.shape = Shape.getReducedShape(i_v.getResultShape(),dimensions);
            f().validateDifferentialFunctionsameDiff(i_v);
            f().validateDifferentialFunctionsameDiff(i_v2);
            addAsNewVertexId();
            f().addFunctionEdges(this);
            this.opState.setAxes(dimensions);


        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

    }



    public BaseAccumulation() {}




    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     * @param n the number of elements
     */
    public BaseAccumulation(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        init();
        //      if (y != null)
        //            LinAlgExceptions.assertSameLength(x, y);
        //LinAlgExceptions.assertSameLength(x, z);

    }

    public BaseAccumulation(INDArray x, INDArray y, long n) {
        this(x, y, x, n);
    }

    public BaseAccumulation(INDArray x) {
        this(x, null, x, x.lengthLong());
    }

    public BaseAccumulation(INDArray x, INDArray y) {
        this(x, y, x, x.lengthLong());
        //if (y != null)
        //    LinAlgExceptions.assertSameLength(x, y);
    }

    public BaseAccumulation(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }




    private void init() {
        if (z == null || x == z)
            init(x, y, x, x.lengthLong());
        else
            init(x, y, z, x.lengthLong());
    }

    @Override
    public INDArray noOp() {
        if (z != null && x != z)
            return z().assign(x);
        else
            return x().dup(x().ordering());
    }


    @Override
    public TOp asIntermediateRepresentation(OnnxProto3.NodeProto node, TGraph graph, Map<String, OnnxProto3.AttributeProto> attributesForNode) {
        return returnIntermediateRpresentation(buildBasicNode(node,graph),graph);
    }


    /**
     * This method returns given TF node as TOp
     *
     * @return
     */
    @Override
    public TOp asIntermediateRepresentation(@NonNull NodeDef node, @NonNull TGraph graph) {
        return returnIntermediateRpresentation(buildBasicNode(node,graph),graph);
    }

    private TOp returnIntermediateRpresentation(TOp tNode,TGraph graph) {
        /**
         * 2 options here. We either have specific dimension, or not.
         * If not - that'll be reduceScalar, if yes - there will be reduceAlongDimension
         */

        log.debug("TOp inputs: {}", tNode.getInputs());
        val shapeIndex = tNode.getInputs().remove(1);

        val variable = graph.getVariableSpace().getVariable(shapeIndex);

        // reduce to scalar
        if (variable.getArray() == null && variable.getShape().length == 2 && variable.getShape()[0] == 1 && variable.getShape()[1] == 1)
            tNode.getOpState().setAxes(new int[]{Integer.MAX_VALUE});// we're going for scalar
        else {
            if (variable.getArray() != null) {
                val axes = variable.getArray().data().asInt();
                tNode.getOpState().setAxes(axes);
            } else
                tNode.getOpState().setAxes(variable.getShape());
        }

        return tNode;
    }

    @Override
    public void setFinalResult(double value) {
        this.finalResult = value;
    }

    @Override
    public Number getFinalResult() {
        return finalResult;
    }

    @Override
    public double zeroDouble() {
        return 0;
    }

    @Override
    public float zeroFloat() {
        return 0;
    }

    @Override
    public float zeroHalf() {
        return 0;
    }


}
