package org.nd4j.linalg.api.ops;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Index based reduction algo
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseIndexAccumulation extends BaseOp implements IndexAccumulation {
    protected int finalResult;


    public BaseIndexAccumulation(SameDiff sameDiff,
                            DifferentialFunction i_v,
                            int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            this.args = new DifferentialFunction[] {i_v};
            this.dimensions = dimensions;
            f().validateDifferentialFunctionsameDiff(i_v);
            addAsNewVertexId();
            f().addFunctionEdges(this);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }

    public BaseIndexAccumulation(SameDiff sameDiff,
                            DifferentialFunction i_v,
                            DifferentialFunction i_v2,
                            int[] dimensions) {
        super(sameDiff,new Object[]{dimensions});
        if (i_v != null) {
            this.args = new DifferentialFunction[] {i_v,i_v2};
            this.dimensions = dimensions;
            f().validateDifferentialFunctionsameDiff(i_v);
            f().validateDifferentialFunctionsameDiff(i_v2);
            addAsNewVertexId();
            f().addFunctionEdges(this);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }
    }


    public BaseIndexAccumulation() {}

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
    public BaseIndexAccumulation(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        init();
    }

    public BaseIndexAccumulation(INDArray x, INDArray y, long n) {
        this(x, y, x, n);
    }

    public BaseIndexAccumulation(INDArray x) {
        this(x, null, x, x.lengthLong());
    }

    public BaseIndexAccumulation(INDArray x, INDArray y) {
        this(x, y, x, x.lengthLong());
    }

    @Override
    public double zeroDouble() {
        return 0.0;
    }

    @Override
    public float zeroFloat() {
        return 0.0f;
    }

    @Override
    public Pair<Double, Integer> zeroPair() {
        return new Pair<>(zeroDouble(), -1);
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0.0, 0.0);
    }

    private void init() {
        init(x, y, x, x.lengthLong());
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            this.extraArgs = new Object[] {zeroDouble()};
        } else if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            this.extraArgs = new Object[] {zeroFloat()};
        } else if (Nd4j.dataType() == DataBuffer.Type.HALF) {
            this.extraArgs = new Object[] {zeroHalf()};
        }
    }


    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>(1);
        ret.add(Shape.getReducedShape(arg().getResultShape(),dimensions));
        return ret;
    }



    @Override
    public void setFinalResult(int idx) {
        this.finalResult = idx;
    }

    @Override
    public int getFinalResult() {
        return finalResult;
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

}
