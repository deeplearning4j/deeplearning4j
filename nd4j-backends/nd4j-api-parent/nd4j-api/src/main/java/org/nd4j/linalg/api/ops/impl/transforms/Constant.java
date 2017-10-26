package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.Arrays;
import java.util.List;

@Data
public class Constant extends BaseTransformOp {

    protected NDArrayInformation m_x;
    protected int[] shape;

    public Constant() {
    }


    protected Constant(SameDiff sameDiff,
                       NDArrayInformation i_v,
                       int[] shape,
                       boolean inPlace,int[] vertexId) {
        super();
        this.shape = shape;
        this.inPlace = inPlace;
        this.sameDiff = sameDiff;
        this.args = new DifferentialFunction[] {this};
        if (i_v != null) {
            m_x = i_v;

        } else {
            throw new IllegalArgumentException("Input not null value.");
        }

        this.vertexId = vertexId;
        validateFunctionReference(this);
        if(sameDiff.getGraph().getVertex(this.vertexId[0]) == null) {
            sameDiff.getGraph().addVertex(new NDArrayVertex(sameDiff,vertexId[0],0,i_v));
        }

    }

    public Constant(SameDiff sameDiff,
                    NDArrayInformation i_v,
                    int[] shape,int[] vertexId) {
        this(sameDiff,i_v,shape,false,vertexId);
    }

    public Constant(INDArray x, INDArray z, int[] shape) {
        super(x, z);
        this.shape = shape;
    }

    public Constant(int[] shape) {
        this.shape = shape;
    }

    public Constant(INDArray x, INDArray z, long n, int[] shape) {
        super(x, z, n);
        this.shape = shape;
    }

    public Constant(INDArray x, INDArray y, INDArray z, long n, int[] shape) {
        super(x, y, z, n);
        this.shape = shape;
    }

    public Constant(INDArray x, int[] shape) {
        super(x);
        this.shape = shape;
    }

    /**
     * Get the result shape for this function
     *
     * @return
     */
    @Override
    public int[] getResultShape() {
        return shape;
    }

    @Override
    public boolean isConstant() {
        return true;
    }


    @Override
    public NDArrayInformation getResult() {
        return this.m_x;
    }

    @Override
    public DifferentialFunction[] args() {
        return new DifferentialFunction[] {this};
    }

    @Override
    public DifferentialFunction arg() {
        return this;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> i_v) {
        validateDifferentialFunctionsameDiff(i_v);
        Zero ret = new Zero(sameDiff,shape,new int[]{sameDiff.graph().nextVertexId()});
        DifferentialFunction add = ret;
        return Arrays.asList(add);
    }

    @Override
    public String toString() {
        return m_x.toString();
    }



    @Override
    public DifferentialFunction dup() {
        Constant ret = sameDiff.setupFunction(new Constant(sameDiff, m_x, shape,vertexId));
        Constant differentialFunction = ret;
        return differentialFunction;
    }



    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "constant";
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        return null;
    }
}
