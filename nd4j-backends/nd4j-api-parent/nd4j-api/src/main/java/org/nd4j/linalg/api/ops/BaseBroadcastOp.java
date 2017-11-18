package org.nd4j.linalg.api.ops;

import com.google.common.base.Preconditions;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@NoArgsConstructor
@Slf4j
public abstract class BaseBroadcastOp extends BaseOp implements BroadcastOp {

    protected int[] dimension;


    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           int[] dimension) {
        this(sameDiff,i_v1,i_v2,false,dimension);
    }

    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           boolean inPlace,
                           int[] dimension) {
        super(sameDiff,inPlace,new Object[] {i_v2});
        if (i_v1 != null && i_v2 != null) {
            this.args = new DifferentialFunction[] {sameDiff.setupFunction(i_v1),sameDiff.setupFunction(i_v2)};
            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);
            f().validateFunctionReference(i_v1);
            f().validateFunctionReference(i_v2);
            this.sameDiff = sameDiff;
            this.inPlace = inPlace;
            this.dimension = dimension;
            this.shape = Shape.getBroadcastDimensions(i_v1.getResultShape(),i_v2.getResultShape());
            addAsNewVertexId();
            f().addFunctionEdges(this);
            this.opState.setAxes(dimension);

        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }

        Preconditions.checkState(sameDiff.setupFunction(this) == this);
    }

    public BaseBroadcastOp(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v1,
                           DifferentialFunction i_v2,
                           int[] dimension,
                           Object[] extraArgs) {
        super(sameDiff,extraArgs);
        this.dimension = dimension;
        if (i_v1 != null && i_v2 != null) {
            this.args = new DifferentialFunction[] {sameDiff.setupFunction(i_v1),sameDiff.setupFunction(i_v2)};

            f().validateDifferentialFunctionsameDiff(i_v1);
            f().validateDifferentialFunctionsameDiff(i_v2);

            this.sameDiff = sameDiff;
            this.shape = Shape.getBroadcastDimensions(i_v1.getResultShape(),i_v2.getResultShape());
            addAsNewVertexId();
            f().addFunctionEdges(this);
            this.opState.setAxes(dimension);


        } else {
            throw new IllegalArgumentException("Input not null variables.");
        }

        Preconditions.checkState(sameDiff.setupFunction(this) == this);

    }




    public BaseBroadcastOp(SameDiff sameDiff,DifferentialFunction i_v,int[] dimension,boolean inPlace) {
        this(sameDiff,i_v,i_v.getResultShape(),inPlace,dimension,null);
    }

    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v,
                           int[] shape,
                           boolean inPlace,
                           int[] dimension,
                           Object[] extraArgs) {
        super(sameDiff,inPlace,extraArgs);
        this.shape = shape;
        this.dimension = dimension;
        if (i_v != null) {
            this.args = new DifferentialFunction[] {sameDiff.setupFunction(i_v)};
            f().validateFunctionReference(i_v);
            f().validateDifferentialFunctionsameDiff(i_v);
            addAsNewVertexId();
            f().addFunctionEdges(this);
            this.opState.setAxes(dimension);

        } else {
            throw new IllegalArgumentException("Input not null variable.");
        }

        Preconditions.checkState(sameDiff.setupFunction(this) == this);

    }


    public BaseBroadcastOp(SameDiff sameDiff,
                           DifferentialFunction i_v,
                           int[] dimension,
                           Object[] extraArgs) {
        this(sameDiff,i_v,i_v.getResultShape(),false,dimension,extraArgs);
    }

    public BaseBroadcastOp(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, x.lengthLong());
        this.dimension = dimension;
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] < 0)
                dimension[i] += x.rank();

    }

    @Override
    public Type opType() {
        return Type.BROADCAST;
    }

    /**
     * Calculate the output shape for this op
     * @return
     */
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>();
        ret.add(Shape.broadcastOutputShape(larg().getResultShape(),rarg().getResultShape()));
        return ret;
    }

    @Override
    public int broadcastLength() {
        if (y == null)
            throw new IllegalStateException("Unable to get broad cast length for y, no y specified");
        return y.length();
    }

    @Override
    public int[] broadcastShape() {
        return calculateOutputShape().get(0);
    }

    @Override
    public int[] getDimension() {
        return dimension;
    }

    @Override
    public void setDimension(int... dimension) {
        this.dimension = dimension;
    }


    @Override
    public TOp asIntermediateRepresentation(OnnxProto3.NodeProto node, TGraph graph, Map<String, OnnxProto3.AttributeProto> attributesForNode) {
        val tNode = buildBasicNode(node, graph);
        return returnIntermediateRpresentation(tNode,graph);
    }

    /**
     * This method returns given TF node as TOp
     *
     * @return
     */
    @Override
    public TOp asIntermediateRepresentation(@NonNull NodeDef node, @NonNull TGraph graph) {
        val tNode = buildBasicNode(node, graph);
        return returnIntermediateRpresentation(tNode,graph);
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
